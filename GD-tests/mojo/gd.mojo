from math import sqrt
from ndarray import NDArray

# fn loss[dtype: DType](x: NDArray[dtype], D: NDArray[dtype]) -> Scalar[dtype]:



# fn compute_gradient[dtype: DType](x: NDArray[dtype], D: NDArray[dtype]) -> NDArray[dtype]:
#     var N = x.shape[0]
#     var M = x.shape[1]
#     var grad = NDArray[dtype](x.shape)
#     grad.zeros()
#     for i in range(N):
#         for j in range(N):
#             var sq_dist


#     return grad


fn compute_gradient_simple[dtype: DType](X: NDArray[dtype], D: NDArray[dtype]) -> NDArray[dtype]:
    var N = X.shape[0]
    var dim = X.shape[1]
    var grad = NDArray[dtype](X.shape)
    grad.zeros()

    for i in range(N):
        for j in range(N):
            var sq_dist = Scalar[dtype](0)
            for d in range(dim):
                sq_dist += (X[i, d] - X[j, d])**2

            var residual = sq_dist - D[i, j] * D[i, j]
            for d in range(dim):
                grad[i, d] += 4 * residual * (X[i, d] - X[j, d])

    return grad


fn gradient_descent[dtype: DType](mut X: NDArray[dtype], D: NDArray[dtype], learning_rate: Scalar[dtype], num_iters: Int) -> NDArray[dtype]:
    for _ in range(num_iters):
        var grad = compute_gradient_simple(X, D)
        X = X - (learning_rate * grad)
    return X


fn generate_distance_matrix[dtype: DType](points: NDArray[dtype]) -> NDArray[dtype]:
    N = points.shape[0]
    dim = points.shape[1]
    var distance: SIMD[dtype, 1]
    var D = NDArray[dtype]([N, N])
    D.zeros()

    for i in range(N):
        for j in range(i+1, N):
            distance = 0
            for d in range(dim):
                distance += (points[j, d] - points[i, d])**2
            distance = sqrt(distance)
            D[i, j] = distance
            D[j, i] = distance

    return D


