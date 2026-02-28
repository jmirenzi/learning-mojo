from math import sqrt
from ndarray import NDArray
from sys.info import simd_width_of

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

fn compute_gradient_vectorized[dtype: DType, dim: Int](X: NDArray[dtype], D: NDArray[dtype]) -> NDArray[dtype]:
    comptime simd_width = simd_width_of[dtype]()
    comptime remainder = dim % simd_width  # compile-time tail width
    var N = X.shape[0]
    var grad = NDArray[dtype](X.shape)
    grad.zeros()

    for i in range(N):
        for j in range(N):
            var xi_base = i * X.strides[0]
            var xj_base = j * X.strides[0]

            # Main SIMD loop
            var sq_dist = SIMD[dtype, simd_width](0)
            var d = 0
            while d + simd_width <= dim:
                var xi = X.data.load[width=simd_width](xi_base + d)
                var xj = X.data.load[width=simd_width](xj_base + d)
                var diff = xi - xj
                sq_dist += diff * diff
                d += simd_width
            var sq_dist_scalar = sq_dist.reduce_add()

            # Tail — only if remainder exists
            @parameter
            if remainder > 0:
                var xi = X.data.load[width=remainder](xi_base + d)
                var xj = X.data.load[width=remainder](xj_base + d)
                var diff = xi - xj
                sq_dist_scalar += (diff * diff).reduce_add()

            var dij = D.data[i * D.strides[0] + j]
            var residual = sq_dist_scalar - dij * dij

            # Gradient accumulation main loop
            var res_vec = SIMD[dtype, simd_width](4 * residual)
            d = 0
            while d + simd_width <= dim:
                var xi = X.data.load[width=simd_width](xi_base + d)
                var xj = X.data.load[width=simd_width](xj_base + d)
                var g  = grad.data.load[width=simd_width](xi_base + d)
                grad.data.store(xi_base + d, g + res_vec * (xi - xj))
                d += simd_width

            # Gradient accumulation tail
            @parameter
            if remainder > 0:
                var res_vec_tail = SIMD[dtype, remainder](4 * residual)
                var xi = X.data.load[width=remainder](xi_base + d)
                var xj = X.data.load[width=remainder](xj_base + d)
                var g  = grad.data.load[width=remainder](xi_base + d)
                grad.data.store(xi_base + d, g + res_vec_tail * (xi - xj))

    return grad

fn compute_gradient_vectorized[dtype: DType](X: NDArray[dtype], D: NDArray[dtype]) -> NDArray[dtype]:
    comptime simd_width = simd_width_of[dtype]()
    var N = X.shape[0]
    var dim = X.shape[1]
    var grad = NDArray[dtype](X.shape)
    grad.zeros()

    for i in range(N):
        for j in range(N):
            # Compute base offsets once, outside the d loop
            var xi_base = i * X.strides[0]
            var xj_base = j * X.strides[0]

            # Vectorize over d to compute squared distance
            var sq_dist = SIMD[dtype, simd_width](0)
            var d = 0
            while d + simd_width <= dim:
                var xi = X.data.load[width=simd_width](xi_base + d)
                var xj = X.data.load[width=simd_width](xj_base + d)
                var diff = xi - xj
                sq_dist += diff * diff
                d += simd_width
            var sq_dist_scalar = sq_dist.reduce_add()
            while d < dim:
                var diff = X.data[xi_base + d] - X.data[xj_base + d]
                sq_dist_scalar += diff * diff
                d += 1

            # D[i,j] — single scalar access
            var dij = D.data[i * D.strides[0] + j]
            var residual = sq_dist_scalar - dij * dij

            # Vectorize over d to accumulate gradient
            var res_vec = SIMD[dtype, simd_width](4 * residual)
            d = 0
            while d + simd_width <= dim:
                var xi = X.data.load[width=simd_width](xi_base + d)
                var xj = X.data.load[width=simd_width](xj_base + d)
                var g = grad.data.load[width=simd_width](xi_base + d)
                grad.data.store(xi_base + d, g + res_vec * (xi - xj))
                d += simd_width
            while d < dim:
                grad.data[xi_base + d] += 4 * residual * (X.data[xi_base + d] - X.data[xj_base + d])
                d += 1

    return grad


fn gradient_descent[dtype: DType,dim: Int](mut X: NDArray[dtype], D: NDArray[dtype], learning_rate: Scalar[dtype], num_iters: Int) -> NDArray[dtype]:
    for _ in range(num_iters):
        var grad = compute_gradient_vectorized[dtype,dim](X, D)
        # var grad = compute_gradient_simple(X, D)
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


