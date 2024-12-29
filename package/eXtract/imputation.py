import numpy as np
from scipy.ndimage import convolve # type: ignore
from scipy.signal import convolve2d # type: ignore

def normalize_adjacency_matrix(A):
    """
    Dodanie własnych pętli i normalizacja symetryczna macierzy sąsiedztwa.
    Args:
        A (numpy.ndarray): Macierz sąsiedztwa (kontaktów).

    Returns:
        numpy.ndarray: Znormalizowana macierz sąsiedztwa.
    """
    # Dodanie własnych pętli (identity matrix)
    A_tilde = A + np.eye(A.shape[0])
    # Wyznaczenie macierzy stopni
    D = np.diag(A_tilde.sum(axis=1))
    # Normalizacja symetryczna: D^(-1/2) * A_tilde * D^(-1/2)
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_normalized = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_normalized

def imputation(contact_matrix, w=5, p=0.85, tol=1e-6, max_iterations=100, threshold_percentile=80):
    # Parameters
    #w = 5       Window size for genomic neighbor-based imputation
    #p = 0.85      Restart probability for RWR
    #tol = 1e-6    Tolerance for convergence in RWR
    #max_iterations = 100   Maximum iterations for RWR
    #threshold_percentile = 80   Percentile for binarization

    #-------- Genomic neighbor-based imputation ----------------------

    kernel_size = 2 * w + 1
    kernel = np.ones((kernel_size, kernel_size))
    M1 = convolve(contact_matrix, kernel, mode='constant', cval=0.0)

    #-------- Random Walk with Restart (RWR)-based imputation --------

    # Normalize M1 row-wise to get transition matrix R
    R = normalize_adjacency_matrix(M1)

    # Initialize M2 as identity matrix
    n = M1.shape[0]
    M2 = np.identity(n)

    # Perform RWR until convergence or maximum iterations reached
    for iteration in range(max_iterations):
        M2_new = p * M2.dot(R) + (1 - p) * np.identity(n)
        if np.linalg.norm(M2_new - M2, ord='fro') < tol:
            M2 = M2_new
            # print(f'Finished at {iteration} iteration')
            break
        M2 = M2_new
    
    #-------- Graph convolution-based imputation ---------------------
    # Define a kernel for immediate neighbors (excluding the center)
    gc_kernel = np.ones((3, 3))
    gc_kernel[1, 1] = 0

    M3 = convolve2d(M2, gc_kernel, mode='same', boundary='fill', fillvalue=0)

    #-------- Normalization of the matrix before binarization --------
    M3 = (M3 - M3.min()) / (M3.max() - M3.min())
    
    # Binarization of the imputed matrix
    threshold = np.percentile(M3, threshold_percentile)
    return (M3 >= threshold).astype(int) # M_binarized