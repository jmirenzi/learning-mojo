from sys.info import simd_width_of, size_of
from bit import next_power_of_two
from random import rand
from math import ceildiv
from memory import alloc, memcpy

struct NDArray[dtype: DType](Stringable,ImplicitlyCopyable):
    var data: UnsafePointer[Scalar[Self.dtype],MutExternalOrigin]
    var shape: List[Int]
    var strides: List[Int]
    var ndim: Int
    var size: Int

    fn __init__(out self, shape: List[Int]):
        self.shape = List[Int]()
        self.ndim = len(shape)
        self.strides = []
        self.size = 1
        # Build strides right-to-left, same as numpy C order
        for i in range(self.ndim):
            self.shape.append(shape[i])
            self.size *= shape[i]

        var stride = 1
        for i in range(self.ndim - 1, -1, -1):
            self.strides.append(stride)
            stride *= shape[i]

        # strides were built back-to-front, reverse to align with shape
        self.strides.reverse()

        comptime simd_width = simd_width_of[Self.dtype]()

        self.size = Int(ceildiv(self.size, simd_width) * simd_width)
        self.data = alloc[Scalar[Self.dtype]](self.size)

    fn __del__(deinit self):
        self.data.free()

    fn __copyinit__(out self, copy: NDArray[Self.dtype]):
        self.data = alloc[Scalar[Self.dtype]](copy.size)
        self.shape = copy.shape.copy()
        self.strides = copy.strides.copy()
        self.ndim = copy.ndim
        self.size = copy.size
        memcpy[Scalar[Self.dtype]](dest=self.data, src=copy.data, count=self.size)

    @always_inline
    fn _flat_index(self, indices: VariadicList[Int]) -> Int:
        var offset = 0
        for i in range(len(indices)):
            offset += indices[i] * self.strides[i]
        return offset

    @always_inline
    fn load[nelts: Int](self, indices: VariadicList[Int]) -> SIMD[Self.dtype, nelts]:
        offset = self._flat_index(indices)
        return self.data.load[nelts](offset)

    @always_inline
    fn store[nelts: Int](mut self, indices: VariadicList[Int], val: SIMD[Self.dtype, nelts]):
        offset = self._flat_index(indices)
        return self.data.store[nelts](offset, val)

    @always_inline
    fn __getitem__(self, *indices: Int) -> SIMD[Self.dtype, 1]:
        return self.load[1](indices)

    @always_inline
    fn __setitem__(mut self, *indices: Int, val: SIMD[Self.dtype, 1]):
        self.store[1](indices, val)

    @always_inline
    fn fill(mut self, value: Scalar[Self.dtype]):
        comptime simd_width = simd_width_of[Self.dtype]()
        var broadcast = SIMD[Self.dtype, simd_width](value)
        for i in range(0, self.size, simd_width):
            self.data.store(i, broadcast)

    @always_inline
    def __add__(mut self, other: NDArray[Self.dtype]):
        # assert self.shape == other.shape
        comptime simd_width = simd_width_of[Self.dtype]()
        for i in range(0, self.size, simd_width):
            var a = self.data.load[simd_width](i)
            var b = other.data.load[simd_width](i)
            self.data.store(i, a + b)

    fn zeros(mut self):
        self.fill(0)

    fn ones(mut self):
        self.fill(1)

    fn identity(mut self):
        self.zeros()
        for i in range(min(self.shape[0], self.shape[1])):
            self[i, i] = SIMD[Self.dtype, 1](1)

    fn random(mut self):
        rand(self.data, self.size)

    fn transpose(mut self):
        # Swap shape and strides for the last two dimensions
        if self.ndim < 2:
            return # No-op for 1D or scalar arrays
        self.shape[self.ndim - 1], self.shape[self.ndim - 2] = self.shape[self.ndim - 2], self.shape[self.ndim - 1]
        self.strides[self.ndim - 1], self.strides[self.ndim - 2] = self.strides[self.ndim - 2], self.strides[self.ndim - 1]

    fn matmut(mut self, other: NDArray[Self.dtype]) -> NDArray[Self.dtype]:
        # Simple matrix multiplication for last two dimensions
        debug_assert(self.ndim == 2 and other.ndim == 2)
        debug_assert(self.shape[:2] == other.shape[:2])
        debug_assert(self.dtype == other.dtype)
        debug_assert(self.shape[-1] == other.shape[-2],"matmul inner dimensions must match")
        comptime simd_width = simd_width_of[Self.dtype]()

        var M = self.shape[self.ndim - 2]
        var K = self.shape[self.ndim - 1]
        var N = other.shape[other.ndim - 1]
        var result_shape = self.shape.copy()
        result_shape[-1] = other.shape[-1]
        var result = NDArray[Self.dtype](result_shape)

        var num_batch = 1
        for d in self.shape[0:self.ndim - 2]:
            num_batch = num_batch * d
        var self_batch_stride = self.strides[-3] if self.ndim > 2 else M*K
        var other_batch_stride = other.strides[-3] if other.ndim > 2 else K*N
        var result_batch_stride = result.strides[-3] if result.ndim > 2 else M*N
        result.zeros()

        for b in range(num_batch):
            var self_offset = b * self_batch_stride
            var other_offset = b * other_batch_stride
            var result_offset = b * result_batch_stride

            for m in range(M):
                for k in range(K):
                    # Broadcast a single self value across the N loop
                    var a_val = SIMD[Self.dtype, simd_width](self.data[self_offset + m * K + k])
                    var r_base = result_offset + m * N
                    var o_base = other_offset  + k * N
                    var n = 0

                    # SIMD main loop: process nelts columns at a time
                    while n + simd_width <= N:
                        var r_vec = result.data.load[width=simd_width](r_base + n)
                        var o_vec = other.data.load[width=simd_width](o_base + n)
                        result.data.store(r_base + n, r_vec + a_val * o_vec)
                        n += simd_width

                    # Scalar remainder
                    while n < N:
                        result.data[r_base + n] += self.data[self_offset + m * K + k] * other.data[o_base + n]
                        n += 1
        return result

    fn __str__(self) -> String:
        return self._format_recursive(0, 0)

    fn _format_recursive(self, dim: Int, offset: Int) -> String:
        # Base case: we're at the last dimension, print actual values
        if dim == self.ndim - 1:
            var result = String("[")
            for i in range(self.shape[dim]):
                result += String(self.data[offset + i * self.strides[dim]])
                if i < self.shape[dim] - 1:
                    result += ", "
            result += "]"
            return result

        # Recursive case: wrap each slice in brackets with indentation
        var result = String("[")
        for i in range(self.shape[dim]):
            var child_offset = offset + i * self.strides[dim]
            # Indent by current depth so nested brackets line up
            if i > 0:
                for _ in range(dim + 1):
                    result += " "
            result += self._format_recursive(dim + 1, child_offset)
            if i < self.shape[dim] - 1:
                # More newlines for higher-level separations, like numpy
                result += ",\n"
        result += "]"
        return result


def main():
    var mat = NDArray[DType.int64]([2,3])
    mat[0, 0] = 1
    mat[0, 1] = 2
    mat[0, 2] = 3
    mat[1, 0] = 4
    mat[1, 1] = 5
    mat[1, 2] = 6

    print(String(mat))

    mat.fill(42)
    print(String(mat))

    mat.identity()
    print(String(mat))

    mat.random()
    print(String(mat))


    var vec = NDArray[DType.float32]([2,3,2])
    vec.random()
    print(String(vec))

    mat1 = NDArray[DType.int32]([2,2,3])
    mat1.random()
    mat2 = NDArray[DType.int32]([2,3,2])
    mat2.random()
    var product = mat1.matmut(mat2)
    print("mat1:\n" + String(mat1))
    print("mat2:\n" + String(mat2))
    print("product:\n" + String(product))