import numpy as np

def compute_insulation_scores(cell: np.ndarray, scale: int) -> np.ndarray:
    """
    Computes the insulation score for each bin in the Hi-C contact matrix.

    Parameters:
    -----------
    cell : np.ndarray
        A square Hi-C contact matrix (N x N), already binned to 1 Mb resolution.
    scale : int
        The number of bins on each side of the current bin that will be considered for summation.
        For example, if scale = 1, the neighborhood [b-1, b+1] is considered in both rows and columns.

    Returns:
    --------
    ins_scores : np.ndarray
        A length-N vector with computed insulation scores for each bin.
    """

    N = cell.shape[0]
    ins_scores = np.zeros(N, dtype=float)

    # We assume the contact matrix is symmetric.
    # Ins(b) = sum of contacts in the window (b-scale : b+scale, b-scale : b+scale),
    # taking into account the matrix boundaries.

    for b in range(N):
        start = max(b - scale, 0)
        end = min(b + scale, N - 1)

        # Extract a submatrix from 'cell'. This submatrix is a square of dimensions [2*scale + 1]
        # (or smaller at the edges).
        submat = cell[start:end+1, start:end+1]

        # Sum all contacts in this subregion:
        ins_scores[b] = np.nansum(submat)  # using nansum in case of any NaN values

    return ins_scores


def compute_insulation_features(cell: np.ndarray, scale: int=100) -> dict:
    """
    Returns a dictionary with several insulation statistics,
    useful for classifying scHi-C data into different cell cycle stages.
    """
    N = cell.shape[0]
    ins_scores = np.zeros(N, dtype=float)

    for b in range(N):
        left_start = max(b - scale, 0)
        left_end   = b
        right_start= b + 1
        right_end  = min(b + scale, N - 1)

        if left_end < left_start or right_end < right_start:
            ins_scores[b] = 0.0
            continue

        # Extract the submatrix between the left region [left_start : left_end]
        # and the right region [right_start : right_end].
        submat = cell[left_start:left_end+1, right_start:right_end+1]
        ins_scores[b] = np.nansum(submat)

    # We now have the ins_scores vector
    avg = np.mean(ins_scores)
    med = np.median(ins_scores)
    std = np.std(ins_scores)

    # np.percentile is good for assessing minima or maxima
    p10 = np.percentile(ins_scores, 10)
    p90 = np.percentile(ins_scores, 90)

    # Return a set of interesting statistics
    return {
        "mean_ins": float(avg),
        "median_ins": float(med),
        "std_ins": float(std),
        "p10_ins": float(p10),
        "p90_ins": float(p90),
        "vector": ins_scores
    }