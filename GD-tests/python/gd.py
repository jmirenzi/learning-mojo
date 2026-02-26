import numpy as np

def loss(x: np.ndarray, D: np.ndarray) -> float:
    N = x.shape[0]
    diffs = np.repeat(x, N, axis=0) - np.tile(x, (N, 1))
    squared_diffs = np.sum(diffs**2, axis=1)

    return np.sum(np.power(squared_diffs - (D**2).flatten(), 2))

def loss_numpy(X, D):
    # 1. Calculate the squared norm of each row: sum(x_i^2)
    # shape: (N, 1)
    sq_norms = np.sum(X**2, axis=1).reshape(-1, 1)

    # 2. Compute all-pairs squared Euclidean distances
    # Formula: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
    # dist_sq shape: (N, N)
    dist_sq = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
    dist_sq = np.maximum(dist_sq, 0)  # Numerical stability: ensure non-negative distances

    # 3. Calculate the squared error against D^2 and sum it up
    return np.sum((dist_sq - D**2)**2)

def compute_gradient_numpy(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    N = X.shape[0]

    # 1. Calculate squared norms
    sq_norms = np.sum(X**2, axis=1).reshape(-1, 1)

    # 2. Compute all-pairs squared Euclidean distances
    dist_sq = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
    dist_sq = np.maximum(dist_sq, 0)

    # 3. Compute errors
    error = dist_sq - D**2  # shape (N, N)

    # 4. Compute differences between all pairs
    X_i = X[:, np.newaxis, :]  # shape (N, 1, dim)
    X_j = X[np.newaxis, :, :]  # shape (1, N, dim)
    diffs = X_i - X_j  # shape (N, N, dim)

    # 5. Compute gradient: grad[i] = 4 * sum_j error[i,j] * (x_i - x_j)
    error_expanded = error[:, :, np.newaxis]  # shape (N, N, 1)
    grad = 4 * np.sum(error_expanded * diffs, axis=1)  # shape (N, dim)

    return grad

def compute_gradient(x: np.ndarray, D: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    dim = x.shape[1]
    diffs = np.repeat(x, N, axis=0) - np.tile(x, (N, 1))
    squared_diffs = np.sum(diffs**2, axis=1)
    error = squared_diffs - (D**2).flatten()

    # Reshape to compute per-point gradients
    diffs = diffs.reshape(N, N, dim)  # Shape: (N, N, dim)
    error = error.reshape(N, N)  # Shape: (N, N)

    # For each point i, sum gradient contributions from all pairs (i,j)
    error_expanded = error[:, :, np.newaxis]  # Shape: (N, N, 1)
    grad = 4 * np.sum(error_expanded * diffs, axis=1)  # Shape: (N, dim)

    return grad

def gradient_descent(x: np.ndarray, D: np.ndarray, learning_rate: float, num_iterations: int) -> np.ndarray:

    for i in range(num_iterations):
        grad = compute_gradient(x, D)
        x -= learning_rate * grad

    return x

def gradient_descent_cache(x: np.ndarray, D: np.ndarray, learning_rate: float, num_iterations: int) -> np.ndarray:
    postion_cache = np.empty([num_iterations] + list(x.shape),dtype=x.dtype)
    loss_cache = np.empty(num_iterations, dtype=x.dtype)

    for i in range(num_iterations):
        postion_cache[i] = x.copy()
        loss_cache[i] = loss(x, D)
        grad = compute_gradient(x, D)
        x -= learning_rate * grad

    return postion_cache, loss_cache