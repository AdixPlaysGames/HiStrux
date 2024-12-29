import numpy as np
from scipy.stats import linregress # type: ignore

def compute_contact_scaling_exponent(contact_matrix: np.ndarray,
                                     min_distance: int = 1,
                                     max_distance: int = None) -> dict:
    """
    Computes the contact scaling exponent from a Hi-C contact matrix (log-log scale).
    Excludes distances for which the average contact p_of_s = 0 from the regression.

    Returns:
    --------
    dictionary :
        slope
        intercept
        r_value
        p_value
        std_err
        distances
        p_of_s
    """

    N = contact_matrix.shape[0]
    if max_distance is None:
        max_distance = N - 1

    sum_of_values = np.zeros(N, dtype=np.float64)
    count_of_values = np.zeros(N, dtype=np.int64)

    # Calculate the sum of contacts and the number of pixels for each distance
    for i in range(N):
        for j in range(i+1, N):
            dist = j - i
            if dist > max_distance:
                break
            val = contact_matrix[i, j]
            if not np.isnan(val):
                sum_of_values[dist] += val
                count_of_values[dist] += 1

    # Compute the average p_of_s
    distances = []
    p_of_s = []
    for s in range(min_distance, max_distance+1):
        if count_of_values[s] > 0:
            mean_val = sum_of_values[s] / count_of_values[s]
            distances.append(s)
            p_of_s.append(mean_val)

    distances = np.array(distances, dtype=float)
    p_of_s = np.array(p_of_s, dtype=float)

    # Filter out distances that have p_of_s = 0
    valid_idx = (p_of_s > 0)
    distances = distances[valid_idx]
    p_of_s = p_of_s[valid_idx]

    # If all values are zero or removed, return NaNs
    if len(p_of_s) == 0:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r_value": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
            "distances": np.array([]),
            "p_of_s": np.array([])
        }

    # Use log10
    log_dist = np.log10(distances)
    log_ps = np.log10(p_of_s)

    # Fit a line in log-log scale
    slope, intercept, r_value, p_value, std_err = linregress(log_dist, log_ps)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err,
        "distances": distances,
        "p_of_s": p_of_s
    }