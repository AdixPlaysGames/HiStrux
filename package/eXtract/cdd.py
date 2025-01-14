import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Spodziewamy się trendu dla wielu komórek zbadamy
# i są nany w wartościach
def compute_cdd(cell: np.ndarray, 
                bin_size: int = 1_000_000,
                exponent_steps: float = 0.33,
                plot: bool = False):
    """
    Computes the contact probability distribution (CDD) depending on genomic distance,
    with an option to adjust logarithmic step size through the 'exponent_steps' parameter.
    It can also optionally plot the resulting distribution if 'plot=True'.

    Parameters
    ----------
    cell : numpy.ndarray
        A 2D contact matrix (N x N), where N is the number of genomic bins.
    bin_size : int, optional
        The size of each genomic bin in base pairs (default is 1,000,000 bp).
    exponent_steps : float, optional
        Determines the logarithmic step size for binning. For example, exponent_steps=0.1
        creates finer-grained bins (10 times denser) compared to exponent_steps=1.0.
    plot : bool, optional
        If True, plots the resulting contact probability distribution (CDD). Defaults to False.

    Returns
    -------
    dict
        A dictionary containing two keys:
        - 'bins_array': List of tuples representing the (start, end) of each log bin in log2 scale.
        - 'probs_array': List of probabilities (normalized contact values) for each bin.
    """

    N = cell.shape[0]
    # Upper-triangle indices (excluding the main diagonal)
    triu_i, triu_j = np.triu_indices(N, k=1)
    # Retrieve the corresponding contact values
    contact_vals = cell[triu_i, triu_j]

    # Filter out zero (or negative) values if any occur
    mask_nonzero = contact_vals > 0
    triu_i = triu_i[mask_nonzero]
    triu_j = triu_j[mask_nonzero]
    contact_vals = contact_vals[mask_nonzero]

    # Calculate genomic distance
    dists = (triu_j - triu_i) * bin_size

    # Compute log2(distance)
    log_vals = np.log2(dists)

    # Determine bin_start and bin_end for log-binning
    bin_starts = np.floor(log_vals / exponent_steps) * exponent_steps
    bin_ends = bin_starts + exponent_steps

    # Aggregate the sum of contact values in each log-bin "drawer"
    dist_counts = defaultdict(float)
    dist_sums = float(np.sum(contact_vals))  # total number of contacts

    for bs, be, cval in zip(bin_starts, bin_ends, contact_vals):
        dist_counts[(bs, be)] += cval

    # Convert counts to probabilities
    bins_array = sorted(dist_counts.keys())  # keys sorted by bin_start
    probs_array = [dist_counts[b] / dist_sums for b in bins_array]

    # Optional plotting
    if plot:
        # For each bin, find the midpoint in log2 space
        # then convert that midpoint back to linear space for plotting.
        bin_mid_log2 = [(bs + be) / 2.0 for (bs, be) in bins_array]
        bin_mid_linear = [2 ** v for v in bin_mid_log2]

        plt.figure(figsize=(7, 5))
        # We can plot on a log-log scale:
        plt.loglog(bin_mid_linear, probs_array, marker='o', linestyle='--', color='blue')
        plt.xlabel("Genomic distance [bp] (log scale)")
        plt.ylabel("Contact probability (log scale)")
        plt.title("Contact Probability Distribution (CDD)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return {
        "bins_array": bins_array,
        "probs_array": probs_array
    }

# tutaj też cały array brać