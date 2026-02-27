import benchmark
from python import Python
from math import cos, sin, pi

from ndarray import NDArray
from gd import compute_gradient_simple, generate_distance_matrix, gradient_descent

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
    return points

@always_inline
fn bechmarking[dtype:DType](mut X: NDArray[dtype], D: NDArray[dtype], learning_rate: Scalar[dtype], num_iters: Int) raises:

    @parameter
    fn test_fun():
        _ = gradient_descent[dtype](X, D, learning_rate, num_iters)

    var report = benchmark.run[test_fun]()
    # Prevent the matrices from being freed before the benchmark run
    _ = (X, D)

    report.print()

fn main():
    N = 100
    comptime dtype = DType.float32
    comptime dim = 2
    var X = generate_radial_points[dtype](N, dim)
    D = generate_distance_matrix(X)
    comptime lr = 0.001
    comptime niter = 1000

    try:
        bechmarking(X, D, learning_rate=lr, num_iters=niter)
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


