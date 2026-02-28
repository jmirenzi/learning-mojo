# main.mojo - Benchmarking gradient descent implementations for MDS problem
import benchmark
from python import Python
from math import cos, sin, pi,sqrt

from ndarray import NDArray
from gd import compute_gradient_simple, generate_distance_matrix, gradient_descent_fast, gradient_descent_simple

fn generate_radial_points[dtype:DType](N: Int, dim: Int) -> NDArray[dtype]:
    comptime assert dtype.is_floating_point()
    var points = NDArray[dtype]([N, dim])
    var radius = Scalar[dtype](1)
    comptime PI = 3.141592653589793
    if dim == 2:
        for i in range(N):
            var angle: Scalar[dtype] = Scalar[dtype](i) / Scalar[dtype](N) * 2 * PI
            points[i, 0] = radius * cos(angle)
            points[i, 1] = radius * sin(angle)
    elif dim == 3:
        for i in range(N):
            var phi: Scalar[dtype] = Scalar[dtype](i) / Scalar[dtype](N) * PI
            var theta: Scalar[dtype] = Scalar[dtype](i) / Scalar[dtype](N) * 2 * PI
            points[i, 0] = radius * sin(phi) * cos(theta)
            points[i, 1] = radius * sin(phi) * sin(theta)
            points[i, 2] = radius * cos(phi)
    else:
        points.random()
        for i in range(N):
            var norm = Scalar[dtype](0)
            for d in range(dim):
                norm += points[i, d] * points[i, d]
            norm = sqrt(norm)
            for d in range(dim):
                points[i, d] = radius * points[i, d] / norm
    return points

@always_inline
fn bechmarking[dtype:DType, dim: Int](mut X: NDArray[dtype], D: NDArray[dtype], learning_rate: Scalar[dtype], num_iters: Int) raises:
    comptime NUM_PARALLEL: Int = 12
    @parameter
    fn test_fun_simple():
        _ = gradient_descent_simple[dtype](X, D, learning_rate, num_iters)
    @parameter
    fn test_fun():
        _ = gradient_descent_fast[dtype,dim,NUM_PARALLEL](X, D, learning_rate, num_iters)

    # print("Running simple gradient descent benchmark...")
    # var report_simple = benchmark.run[test_fun_simple]()
    # report_simple.print()
    print("Running parameterized vectorized gradient descent benchmark with num_parallel=", NUM_PARALLEL, "...")
    var report_parameterized = benchmark.run[test_fun]()
    # Prevent the matrices from being freed before the benchmark run
    _ = (X, D)

    report_parameterized.print()

fn main():
    N = 100
    comptime dtype = DType.float32
    comptime dim = 3
    var X = generate_radial_points[dtype](N, dim)
    D = generate_distance_matrix(X)
    comptime lr = 0.001
    comptime niter = 1000

    try:
        bechmarking[dtype, dim](X, D, learning_rate=lr, num_iters=niter)
    except e:
        print("Error during benchmarking:", e)


    # try:
    #     Python.add_to_path(".")
    #     main_py = Python.import_module("main")
    #     _ = pymain.benchmarks(
    #             D.to_python(),
    #             dim,
    #             lr,
    #             niter,
    #             plots
    #         )
    # except e:
    #     print("Error running Python benchmarks:", e)


