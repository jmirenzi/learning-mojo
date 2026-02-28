from math import sqrt
from sys.info import simd_width_of
from algorithm import parallelize
from ndarray import NDArray


fn generate_distance_matrix[dtype: DType](points: NDArray[dtype]) -> NDArray[dtype]:
    var N = points.shape[0]
    var dim = points.shape[1]
    var D = NDArray[dtype]([N, N])
    D.zeros()
    for i in range(N):
        for j in range(i + 1, N):
            var dist = Scalar[dtype](0)
            for d in range(dim):
                var diff = points[j, d] - points[i, d]
                dist += diff * diff
            dist = sqrt(dist)
            D[i, j] = dist
            D[j, i] = dist
    return D


# ── simple scalar version ────────────────────────────────────────────────────

fn compute_gradient_simple[dtype: DType](
    X: NDArray[dtype], D: NDArray[dtype]
) -> NDArray[dtype]:
    var N = X.shape[0]
    var dim = X.shape[1]
    var grad = NDArray[dtype](X.shape)
    grad.zeros()
    for i in range(N):
        for j in range(N):
            var sq_dist = Scalar[dtype](0)
            for d in range(dim):
                sq_dist += (X[i, d] - X[j, d]) ** 2
            var residual = sq_dist - D[i, j] * D[i, j]
            for d in range(dim):
                grad[i, d] += 4 * residual * (X[i, d] - X[j, d])
    return grad


fn gradient_descent_simple[dtype: DType](
    mut X: NDArray[dtype],
    D: NDArray[dtype],
    learning_rate: Scalar[dtype],
    num_iters: Int,
):
    comptime simd_width = simd_width_of[dtype]()

    for _ in range(num_iters):
        grad = compute_gradient_simple[dtype](X, D)
        X -= learning_rate * grad


# ── SIMD + parallel version ──────────────────────────────────────────────────

fn _compute_gradient_into[dtype: DType, dim: Int, num_parallel: Int](
    X: NDArray[dtype], D: NDArray[dtype], mut grad: NDArray[dtype]
):
    """
    Writes gradient directly into an existing buffer.
    No allocation — caller owns grad.
    dim is a compile-time parameter so SIMD widths are resolved at compile time.
    """
    comptime simd_width = simd_width_of[dtype]()
    comptime remainder = dim % simd_width
    var N = X.shape[0]
    var x_row_stride = X.strides[0]   # == dim, but read from strides for correctness
    var d_row_stride = D.strides[0]   # == N

    @parameter
    fn calc_row(i: Int):
        var xi_base = i * x_row_stride

        for j in range(N):
            var xj_base = j * x_row_stride

            # ── squared distance ─────────────────────────────────────────────
            var sq_dist = SIMD[dtype, simd_width](0)
            var d = 0
            while d + simd_width <= dim:
                var diff = (X.data.load[width=simd_width](xi_base + d) - X.data.load[width=simd_width](xj_base + d))
                sq_dist += diff * diff
                d += simd_width
            var sq_dist_s = sq_dist.reduce_add()

            @parameter
            if remainder > 0:
                var diff = (X.data.load[width=remainder](xi_base + d) - X.data.load[width=remainder](xj_base + d))
                sq_dist_s += (diff * diff).reduce_add()

            # ── residual  4*(||xi-xj||² - D[i,j]²) ─────────────────────────
            var dij = D.data[i * d_row_stride + j]
            var residual4 = Scalar[dtype](4) * (sq_dist_s - dij * dij)

            # ── gradient accumulation ────────────────────────────────────────
            var res_vec = SIMD[dtype, simd_width](residual4)
            d = 0
            while d + simd_width <= dim:
                var xi   = X.data.load[width=simd_width](xi_base + d)
                var xj   = X.data.load[width=simd_width](xj_base + d)
                var g    = grad.data.load[width=simd_width](xi_base + d)
                grad.data.store(xi_base + d, g + res_vec * (xi - xj))
                d += simd_width

            @parameter
            if remainder > 0:
                var res_rem = SIMD[dtype, remainder](residual4)
                var xi  = X.data.load[width=remainder](xi_base + d)
                var xj  = X.data.load[width=remainder](xj_base + d)
                var g   = grad.data.load[width=remainder](xi_base + d)
                grad.data.store(xi_base + d, g + res_rem * (xi - xj))

    # Each row i of grad is independent — safe to parallelize
    parallelize[calc_row](N, num_parallel)


fn gradient_descent_fast[dtype: DType, dim: Int, num_parallel: Int](
    mut X: NDArray[dtype],
    D: NDArray[dtype],
    learning_rate: Scalar[dtype],
    num_iters: Int,
):
    """
    Full optimized GD loop:
      - compile-time dim → SIMD widths resolved at compile time
      - grad buffer allocated once, reused every iteration
      - parallelize over rows in compute_gradient
      - in-place X update, no per-iteration allocation
    """
    comptime simd_width = simd_width_of[dtype]()
    var lr_vec = SIMD[dtype, simd_width](learning_rate)

    # Single allocation for grad — reused every iteration
    var grad = NDArray[dtype](X.shape)

    for _ in range(num_iters):
        grad.zeros()
        _compute_gradient_into[dtype, dim, num_parallel](X, D, grad)

        # X -= lr * grad  in-place, SIMD over the flat buffer
        var i = 0
        while i + simd_width <= X._alloc_size:
            X.data.store(
                i,
                X.data.load[width=simd_width](i)
                - lr_vec * grad.data.load[width=simd_width](i),
            )
            i += simd_width