import numpy as np
from scipy.ndimage import convolve  # type: ignore
from scipy.signal import convolve2d  # type: ignore

def normalize_adjacency_matrix(A: np.ndarray):
    """
    Adds self-loops (identity matrix) and symmetrically normalizes the adjacency matrix.
    
    Args:
        A (numpy.ndarray): Adjacency (contact) matrix.

    Returns:
        numpy.ndarray: Symmetrically normalized adjacency matrix.
    """
    # Add self-loops (identity matrix)
    A_tilde = A + np.eye(A.shape[0])
    # Compute the degree matrix
    D = np.diag(A_tilde.sum(axis=1))
    # Symmetric normalization: D^(-1/2) * A_tilde * D^(-1/2)
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_normalized = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_normalized
    
def imputation(contact_matrix: np.ndarray,
               w: int=5,
               p: float=0.85,
               tol=1e-6,
               max_iterations: int=100,
               threshold_percentile: int=80) -> np.ndarray :
    # Parameters:
    # w: Window size for genomic neighbor-based imputation
    # p: Restart probability for the Random Walk with Restart (RWR)
    # tol: Convergence tolerance for RWR
    # max_iterations: Maximum RWR iterations
    # threshold_percentile: Percentile threshold for binarization

    # -------- Genomic neighbor-based imputation --------
    kernel_size = 2 * w + 1
    kernel = np.ones((kernel_size, kernel_size))
    M1 = convolve(contact_matrix, kernel, mode='constant', cval=0.0)

    # -------- Random Walk with Restart (RWR)-based imputation --------
    # Normalize M1 row-wise to obtain the transition matrix R
    R = normalize_adjacency_matrix(M1)

    # Initialize M2 as the identity matrix
    n = M1.shape[0]
    M2 = np.identity(n)

    # Perform RWR until convergence or until reaching the maximum number of iterations
    for iteration in range(max_iterations):
        M2_new = p * M2.dot(R) + (1 - p) * np.identity(n)
        if np.linalg.norm(M2_new - M2, ord='fro') < tol:
            M2 = M2_new
            # Uncomment the following line to see the iteration at which convergence occurs:
            # print(f'Converged at iteration {iteration}')
            break
        M2 = M2_new
    
    # -------- Graph convolution-based imputation --------
    # Define a kernel for immediate neighbors (excluding the center cell)
    gc_kernel = np.ones((3, 3))
    gc_kernel[1, 1] = 0

    M3 = convolve2d(M2, gc_kernel, mode='same', boundary='fill', fillvalue=0)

    # -------- Normalize the matrix before binarization --------
    M3 = (M3 - M3.min()) / (M3.max() - M3.min())
    
    # Binarize the imputed matrix based on the specified percentile threshold
    threshold = np.percentile(M3, threshold_percentile)
    return (M3 >= threshold).astype(int)  # M_binarized