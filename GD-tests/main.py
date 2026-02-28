# main.py - Benchmarking and visualization of gradient descent implementations for MDS problem
from unittest import result

import numpy as np

from python.gd import gradient_descent, gradient_descent_cache
from python.gd_jax import gradient_descent_JAX, gradient_descent_cache_JAX
from python.viz import plot_gradient_descent, plot_gradient_descent_2D, animate_gradient_descent

from timeit import timeit


def generate_radial_points(N, dim):
    r = 0.5
    points = []
    if dim == 2:
        for i in range(N):
            angle = 2 * np.pi * i / N
            points.append([r * np.cos(angle), r * np.sin(angle)])
    elif dim == 3:
        for i in range(N):
            phi = np.arccos(1 - 2 * (i / N))
            theta = np.sqrt(N * np.pi) * phi
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points.append([x, y, z])
    else:
        points = np.random.rand(N, dim)
        for i in range(N):
            norm = np.linalg.norm(points[i])
            points[i] = r * points[i] / norm

    return points


def generate_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


NUM_ITERS = 10
def benchmark_gradient_descent_JAX(X, D, lr, niter):
    result = gradient_descent_JAX(X, D, learning_rate=lr, num_iterations=niter)
    result.block_until_ready()
    secs = timeit(lambda: gradient_descent_JAX(X, D, learning_rate=lr, num_iterations=niter), number=NUM_ITERS) / NUM_ITERS
    print(f"Average time JAX: {secs}")

def benchmark_gradient_descent(X, D, lr, niter):
    secs = timeit(lambda: gradient_descent(X, D, learning_rate=lr, num_iterations=niter), number=NUM_ITERS) / NUM_ITERS
    print(f"Average time python numpy: {secs}")



def benchmarks(D, dim, lr, niter, plots=True):

    N = len(D)
    D = np.array(D, dtype=np.float64)
    # D_native = PyMatrix(D.tolist(), N, N)

    # Initial starting point
    np.random.seed(42)
    X = np.random.rand(N, dim)
    # X_native = PyMatrix(X.tolist(), N, dim)

    ### Without visuals

    ### Benchmarks
    benchmark_gradient_descent_JAX(X.copy(), D, lr=lr, niter=niter)
    benchmark_gradient_descent(X.copy(), D, lr=lr, niter=niter)

    ## Visualization
    if plots:
        P, L = gradient_descent_cache(X.copy(), D, learning_rate=lr, num_iterations=niter)
        plot_gradient_descent_2D(P, L, title="Gradient Descent in python numpy")
        plot_gradient_descent(P, L, title="Gradient Descent in python numpy")

        # P_native, L_native = gradient_descent_native_cache(X_native.copy(), D_native, learning_rate=lr, num_iterations=niter)
        # plot_gradient_descent(P_native, L_native, title="Gradient Descent in native python")

        if np.allclose(P[0], P[-1]):
            print("Did not move!")
        animate_gradient_descent(P, L, trace=False)


if __name__ == "__main__":

    # Create optimization target
    n_circle = 100
    dim_circle = 3
    points = generate_radial_points(n_circle, dim_circle)           # circle/sphere
    # points = np.loadtxt("./shapes/modular.csv", delimiter=",")      # modular (N = 1000)
    # points = np.loadtxt("./shapes/flame.csv", delimiter=",")        # flame (N = 307)
    print(f"Generated {len(points)} points in {points.shape[1]} dimensions.")
    # Optimization input
    dim = dim_circle
    lr = 0.001
    niter = 1000
    plots = False

    benchmarks(
        D=generate_distance_matrix(points),
        dim=dim,
        lr=lr,
        niter=niter,
        plots=plots
    )