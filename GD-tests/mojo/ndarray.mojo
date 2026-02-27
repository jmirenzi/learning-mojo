from sys.info import simd_width_of
from bit import next_power_of_two
from random import rand
from math import ceildiv
from memory import alloc

struct NDArray[dtype: DType](Stringable):
    var data: UnsafePointer[Scalar[Self.dtype],MutExternalOrigin]
    var shape: List[Int]
    var strides: List[Int]
    var ndim: Int
    var size: Int

    fn __init__(out self, *shape: Int):
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

        var size_near_simd : Int = Int(ceildiv(self.size, simd_width) * simd_width)
        self.data = alloc[Scalar[Self.dtype]](size_near_simd)

    fn __del__(deinit self):
        self.data.free()

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
    var mat = NDArray[DType.int64](2,3)
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


    var vec = NDArray[DType.float32](4,4,4)
    vec.random()
    print(String(vec))