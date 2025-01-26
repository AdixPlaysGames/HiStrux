import numpy as np
from scipy.stats import linregress  # type: ignore
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # type: ignore


def normalize_contact_matrix_vc(contact_matrix: np.ndarray) -> np.ndarray:
    """
    Perform Vanilla Coverage (VC) normalization on a contact matrix.

    Parameters:
    ----------
    contact_matrix : np.ndarray
        A 2D contact matrix to be normalized.
    Returns:
    -------
    np.ndarray
        The normalized contact matrix.
    """
    row_sums = np.sum(contact_matrix, axis=1, keepdims=True)
    col_sums = np.sum(contact_matrix, axis=0, keepdims=True)
    row_sums[row_sums == 0] = 1
    col_sums[col_sums == 0] = 1
    normalized_matrix = contact_matrix / np.sqrt(np.abs(row_sums * col_sums))

    return normalized_matrix


def compute_contact_scaling_exponent(contact_matrix: np.ndarray,
                                     min_distance: int = 1,
                                     max_distance: int = None,
                                     plot: bool = False,
                                     num_log_bins: int = 20) -> dict:
    """
    Computes the contact scaling exponent from a single-cell Hi-C (scHi-C)
    contact matrix in log-log scale with log-binning of distances.

    This function also performs a basic normalization on the contact matrix
    (total sum scaling), because we assume the matrix might not be properly
    normalized beforehand.

    Parameters
    ----------
    contact_matrix : np.ndarray
        A 2D scHi-C contact matrix.
    min_distance : int, optional
        The minimal distance (in bins) to be included in the analysis.
        Default it 1.
    max_distance : int, optional
        The maximal distance (in bins) to be included in the analysis.
        If None, it is set to N-1, where N is the shape of the contact_matrix.
    plot : bool, optional
        Whether to plot the distance vs. p(s) relationship in log-log scale.
        Default is False.
    num_log_bins : int, optional
        How many log-spaced bins to use for log-binning. Default is 20.

    Returns
    -------
    dict
        A dictionary containing:
        - pofs_slope (float)
        - pofs_intercept (float)
        - pofs_r_value (float)
        - pofs_p_value (float)
        - pofs_std_err (float)
        - pofs_distances (np.ndarray)
        - p_of_s (np.ndarray)

    Notes
    -----
    - Distances for which the average contact p(s) = 0 are excluded from
      the linear regression.
    - If plot=True, a log-log plot of distances vs. p(s) is displayed, 
      including the best-fit line based on linear regression 
      in log10-transformed coordinates.
    - Log-binning reduces the scatter of points at large distances,
      where the number of contacts is limited.
    """

    contact_matrix = normalize_contact_matrix_vc(contact_matrix)

    N = contact_matrix.shape[0]
    if max_distance is None:
        max_distance = N - 1

    i_upper, j_upper = np.triu_indices(N, k=1)
    dist = j_upper - i_upper

    # Only consider distances <= max_distance
    valid_mask = (dist <= max_distance) & (dist >= min_distance)
    dist = dist[valid_mask]
    contacts = contact_matrix[i_upper[valid_mask], j_upper[valid_mask]]

    # LOG-BINNING
    bin_edges = np.logspace(
        np.log10(min_distance),
        np.log10(max_distance),
        num_log_bins + 1
    )

    # Determine which bin each distance belongs to
    bin_idx = np.digitize(dist, bin_edges) - 1
    sum_of_values = np.zeros(num_log_bins, dtype=np.float64)
    count_of_values = np.zeros(num_log_bins, dtype=np.int64)

    for i, b in enumerate(bin_idx):
        if 0 <= b < num_log_bins:
            sum_of_values[b] += contacts[i]
            count_of_values[b] += 1

    # Compute the center of each bin (in linear scale)
    bin_centers = 10 ** (
        0.5 * (
            np.log10(bin_edges[:-1]) + np.log10(bin_edges[1:])
        )
    )

    # Calculate p(s) in each bin
    p_of_s_binned = np.zeros(num_log_bins, dtype=np.float64)
    nonzero_mask = count_of_values > 0
    p_of_s_binned[nonzero_mask] = (
        sum_of_values[nonzero_mask] / count_of_values[nonzero_mask]
    )

    distances = bin_centers[nonzero_mask]
    p_of_s = p_of_s_binned[nonzero_mask]

    # smoothing of p_of_s
    p_of_s = gaussian_filter1d(p_of_s, sigma=2)
    valid_idx = (p_of_s > 0)
    distances = distances[valid_idx]
    p_of_s = p_of_s[valid_idx]

    if len(p_of_s) == 0:
        results = {
            "pofs_slope": np.nan,
            "pofs_intercept": np.nan,
            "pofs_r_value": np.nan,
            "pofs_p_value": np.nan,
            "pofs_std_err": np.nan,
            "pofs_distances": np.array([]),
            "p_of_s": np.array([])
        }
        if plot:
            print("No valid p(s) values to plot.")
        return results

    log_dist = np.log10(distances)
    log_ps = np.log10(p_of_s)

    slope, intercept, r_value, p_value, std_err = linregress(log_dist, log_ps)
    results = {
        "pofs_slope": slope,
        "pofs_intercept": intercept,
        "pofs_r_value": r_value,
        "pofs_p_value": p_value,
        "pofs_std_err": std_err,
        "pofs_distances": distances,
        "p_of_s": p_of_s
    }

    if plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(distances, p_of_s, 'o', label='p(s) data', alpha=0.6)
        x_fit = np.linspace(distances.min(), distances.max(), 200)
        log_fit = slope * np.log10(x_fit) + intercept
        y_fit = 10 ** log_fit
        ax.loglog(x_fit, y_fit, 'r-', label=f'Fit: slope={slope:.3f}')

        ax.set_xlabel('Genomic distance (bins) [log-binned]')
        ax.set_ylabel('p(s)')
        ax.set_title('Contact Probability vs. Distance (log-log)')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)

        plt.tight_layout()
        plt.show()

    return results