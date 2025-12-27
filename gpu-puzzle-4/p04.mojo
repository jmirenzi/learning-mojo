from memory import UnsafePointer
from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from testing import assert_equal

from layout import Layout, LayoutTensor

# ANCHOR: add_10_2d
alias SIZE = 2
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)


fn add_10_2d_tensor(
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=True, dtype, layout],
    size: Int,
):
    row = thread_idx.y
    col = thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[row, col] = a[row, col] + 10.0



fn add_10_2d(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    row = thread_idx.y
    col = thread_idx.x
    i = row * size + col
    if row < size and col < size:
        output[i] = a[i] + 10.0


# ANCHOR_END: add_10_2d


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)
        out_2 = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)
        out_2_tensor = LayoutTensor[dtype, layout](out_2)
        expected = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)
        with a.map_to_host() as a_host:
            # row-major
            for i in range(SIZE):
                for j in range(SIZE):
                    a_host[i * SIZE + j] = i * SIZE + j
                    expected[i * SIZE + j] = a_host[i * SIZE + j] + 10
        a_tensor = LayoutTensor[dtype, layout](a)

        ctx.enqueue_function[add_10_2d](
            out.unsafe_ptr(),
            a.unsafe_ptr(),
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        ctx.enqueue_function[add_10_2d_tensor](
            out_2_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        ctx.synchronize()

        print("Raw memory version:")
        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(out_host[i * SIZE + j], expected[i * SIZE + j])

        print("LayoutTensor version:")
        with out_2.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(out_host[i * SIZE + j], expected[i * SIZE + j])