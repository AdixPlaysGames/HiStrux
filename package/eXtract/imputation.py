import numpy as np
from scipy.ndimage import convolve  # type: ignore
from scipy.signal import convolve2d  # type: ignore

def normalize_adjacency_matrix(A: np.ndarray) -> np.ndarray:
    """
    Adds self-loops (identity) and performs symmetric normalization of matrix A.
    Returns a symmetrically normalized matrix.
    """
    # Average A and A.T in case A is not symmetric:
    A = (A + A.T) / 2.0
    
    # Add self-loops (if needed)
    A_tilde = A + np.eye(A.shape[0])
    
    # Degrees
    D = np.diag(A_tilde.sum(axis=1))
    
    # D^(-1/2), avoiding division by zero:
    with np.errstate(divide='ignore', invalid='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(np.diag(D))
        d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Symmetric normalization
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    # Re-enforce symmetry:
    A_norm = (A_norm + A_norm.T) / 2.0
    return A_norm

def imputation(
    contact_matrix: np.ndarray,
    w: int = 5,
    p: float = 0.85,
    tol: float = 1e-6,
    max_iterations: int = 100,
    diagonal_damp: float = 0.2,
    clip_percentile: float = 99.9,
    threshold: float = 0.05
) -> np.ndarray:
    """
    Pipeline for imputing and visualizing scHi-C contact matrices:
      1) Genomic neighborhood convolution
      2) Random Walk with Restart (RWR)
      3) Graph convolution (3x3 excluding center)
      4) (Optional) Diagonal damping
      5) Logarithmic transformation and min-max scaling
      6) (Optional) Clipping at a percentile (e.g., 99.9)
      
    Returns a continuous matrix (float) so that weak contacts outside the diagonal
    are not binarized to 0/1 in visualization.
    
    Args:
        contact_matrix: raw scHi-C contact matrix (can be sparse).
        w: window size for genomic neighborhood convolution.
        p: restart probability in RWR.
        tol: convergence tolerance for RWR (Frobenius norm).
        max_iterations: maximum number of RWR iterations.
        diagonal_damp: damping factor for the diagonal
                       (e.g., 0.0 – no change, 0.5 – half, 1.0 – zero diagonal).
        clip_percentile: percentile for upper clipping after log-transform (default 99.9).
    
    Returns:
        float matrix (NxN) approximately in the range [0,1],
        symmetric, with preserved subtle intensity differences.
    """
    # ---------------- 1. Genomic neighbor-based imputation ----------------
    kernel_size = 2 * w + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    M1 = convolve(contact_matrix, kernel, mode='constant', cval=0.0)
    
    # Enforce symmetry
    M1 = (M1 + M1.T) / 2.0

    # ---------------- 2. Random Walk with Restart (RWR) ----------------
    R = normalize_adjacency_matrix(M1)
    n = R.shape[0]
    M2 = np.eye(n, dtype=float)

    for _ in range(max_iterations):
        M2_new = p * (M2 @ R) + (1 - p) * np.identity(n)
        M2_new = (M2_new + M2_new.T) / 2.0  # enforce symmetry
        if np.linalg.norm(M2_new - M2, ord='fro') < tol:
            M2 = M2_new
            break
        M2 = M2_new

    # ---------------- 3. Graph convolution-based imputation ----------------
    gc_kernel = np.ones((3, 3), dtype=float)
    gc_kernel[1, 1] = 0  # exclude center
    M3 = convolve2d(M2, gc_kernel, mode='same', boundary='fill', fillvalue=0)

    # Enforce symmetry
    M3 = (M3 + M3.T) / 2.0

    # ---------------- 4. (Optional) Diagonal damping ----------------
    # If diagonal_damp > 0, reduce values on the diagonal
    if diagonal_damp > 0:
        diag_indices = np.diag_indices(n)
        M3[diag_indices] = (1.0 - diagonal_damp) * M3[diag_indices]

    # ---------------- 5. Logarithmic transformation + min-max scaling ----------------
    # Ensure M3 is non-negative:
    M3[M3 < 0] = 0.0
    
    # log(1 + x) to emphasize small values
    M3_log = np.log1p(M3)

    # min-max scaling (for log-transform)
    min_val, max_val = M3_log.min(), M3_log.max()
    if max_val > min_val:
        M3_norm = (M3_log - min_val) / (max_val - min_val)
    else:
        # if the matrix is constant (unlikely) – return ones
        return np.ones_like(M3)

    # ---------------- 6. Clipping at the top (optional) ----------------
    # High values (often on the diagonal) can "dim" the rest of the map.
    # Clip at, for example, the 99.9th percentile to better highlight differences in smaller values.
    if clip_percentile is not None and clip_percentile < 100:
        clip_val = np.percentile(M3_norm, clip_percentile)
        M3_norm = np.clip(M3_norm, 0, clip_val)
        # After clipping, we can rescale to [0,1] to maintain dynamics
        min_val2, max_val2 = M3_norm.min(), M3_norm.max()
        if max_val2 > min_val2:
            M3_norm = (M3_norm - min_val2) / (max_val2 - min_val2)

    # Final result is symmetric (ensure it):
    M3_norm = (M3_norm + M3_norm.T) / 2.0
    
    return (M3_norm >= threshold).astype(int)