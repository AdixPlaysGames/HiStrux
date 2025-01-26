import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .imputation import imputation
from scipy.stats import gmean # type: ignore

def compute_directionality_index(contact_matrix, window=5):
    """
    Computes a simplified version of the directionality index (DI).
    """
    N = contact_matrix.shape[0]
    di_values = np.zeros(N, dtype=float)

    # For each bin 'i', define left and right windows. 
    # Compute the sum of contacts on the left side and on the right side, 
    # then use a ratio to determine the directionality index.
    for i in range(N):
        left_start = max(i - window, 0)
        left_end   = i
        right_start = i + 1
        right_end   = min(i + window + 1, N)

        left_sum = contact_matrix[left_start:left_end, i].sum()
        right_sum = contact_matrix[right_start:right_end, i].sum()

        numerator = (right_sum - left_sum)
        denominator = abs(left_sum + right_sum)
        if denominator == 0:
            di = 0
        else:
            di = numerator / denominator
        di_values[i] = di

    return di_values

def detect_tad_boundaries(di_values, threshold=0.8):
    """
    A very simplified TAD boundary detection based on the directionality index (DI).
    """
    boundaries = []
    prev_sign = np.sign(di_values[0])

    # Traverse the DI array and look for sign changes that exceed the specified threshold.
    for i in range(1, len(di_values)):
        cur_sign = np.sign(di_values[i])
        if cur_sign != prev_sign and abs(di_values[i]) > threshold:
            boundaries.append(i)
        prev_sign = cur_sign

    return boundaries

def build_tad_df(chrom, boundaries, n_bins, bin_size):
    """
    From a list of boundaries, build a DataFrame containing:
      [chromosome, tad_id, start_bin, end_bin, start_bp, end_bp, size_in_bins]
    """
    tads = []
    start_bin = 0
    tad_id = 0

    # Each boundary indicates the start of a new TAD, 
    # so create TAD segments between consecutive boundaries.
    for b in boundaries:
        end_bin = b - 1
        if end_bin < start_bin:
            continue
        tads.append((chrom, tad_id, start_bin, end_bin))
        start_bin = b
        tad_id += 1

    if start_bin < n_bins:
        tads.append((chrom, tad_id, start_bin, n_bins - 1))
    data_out = []
    for (chr_, tid, sb, eb) in tads:
        sb_bp = sb * bin_size
        eb_bp = (eb + 1) * bin_size - 1
        size_in_bins = eb - sb + 1
        data_out.append(
            (chr_, tid, sb, eb, sb_bp, eb_bp, size_in_bins)
        )

    tad_df = pd.DataFrame(data_out, columns=[
        "chromosome", "tad_id",
        "start_bin", "end_bin",
        "start_bp", "end_bp",
        "size_in_bins"
    ])
    return tad_df

def compute_tad_stats(tad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes selected TAD statistics for each chromosome.
    """
    if tad_df.empty:
        return pd.DataFrame(columns=["chromosome", "n_tads", "mean_size_in_bins",
                                     "min_size_in_bins", "max_size_in_bins"])

    stats_list = []
    for chrom, group in tad_df.groupby("chromosome"):
        n_tads = len(group)
        mean_size = group["size_in_bins"].mean()
        min_size = group["size_in_bins"].min()
        max_size = group["size_in_bins"].max()

        stats_list.append({
            "chromosome": chrom,
            "n_tads": n_tads,
            "mean_size_in_bins": mean_size,
            "min_size_in_bins": min_size,
            "max_size_in_bins": max_size
        })
    stats_df = pd.DataFrame(stats_list)
    return stats_df

def plot_tads_for_chrom(
    contact_matrix: np.ndarray,
    boundaries: list[int],
    chrom: str,
    out_prefix: str = None,
    show_plot: bool = True
):
    """
    Plots the contact matrix (heatmap) for a given chromosome
    and draws vertical/horizontal lines at TAD boundaries.
    """
    N = contact_matrix.shape[0]
    
    from matplotlib.colors import LinearSegmentedColormap
    custom_blue = LinearSegmentedColormap.from_list("custom_blue", ['#f7f7f7', '#016959'])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.log1p(contact_matrix), 
              cmap=custom_blue, 
              origin='upper', 
              interpolation='nearest')

    for b in boundaries:
        if 0 < b < N:
            ax.axhline(y=b - 0.5, color='blue', linewidth=0.7)
            ax.axvline(x=b - 0.5, color='blue', linewidth=0.7)

    ax.set_title(f"Chrom: {chrom} - TAD boundaries", fontsize=14)
    ax.set_xlabel("Genome position 1", fontsize=12)
    ax.set_ylabel("Genome position 2", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    if out_prefix is not None:
        plt.savefig(f"{out_prefix}_{chrom}_tads.png", dpi=180, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)



def calculate_cis_tads(
    contacts_df: pd.DataFrame,
    bin_size: int = 1_000_000,
    w: int = 5,
    p: float = 0.85,
    imputation_involved: bool = True,
    boundary_threshold: float = 0.8,
    out_prefix: str = None,
    show_plot: bool = False,
    substring: int = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extended version that, after detecting TADs, creates 
    contact matrix plots with TAD boundaries for each chromosome.

    Parameters:
    -----------
    contacts_df : pd.DataFrame
        Input DataFrame with columns like chromosome_1, start_1, ...
    bin_size : int, optional
        Bin size in base pairs, default is 1,000,000.
    w : int, optional
        Window size for directionality index and (optional) imputation, default 5.
    p : float, optional
        Probability for random walk with restart (used in imputation), default 0.85.
    threshold_percentile : int, optional
        Threshold percentile for imputation binarization, default 90.
    imputation_involved : bool, optional
        Whether to apply the imputation step, default False.
    boundary_threshold : float, optional
        Threshold for directionality index sign changes to detect boundaries, default 0.8.
    out_prefix : str, optional
        Prefix for saving output plots as PNG files; if None, no files are saved.
    show_plot : bool, optional
        If True, displays plots using plt.show(); if False, plots are not displayed.
    substring : int, optional
        Number of characters to trim from the end of chromosome names, default None.

    Returns:
    --------
    dict
        A dictionary with:
          "stats_df" : pd.DataFrame containing TAD statistics,
          "tad_df"   : pd.DataFrame with all identified TADs.
    """

    if substring is not None:
        contacts_df['chromosome_1'] = contacts_df['chromosome_1'].str[:-substring]
        contacts_df['chromosome_2'] = contacts_df['chromosome_2'].str[:-substring]
    
    # Filter for cis contacts (same chromosome)
    cis_df = contacts_df[contacts_df["chromosome_1"] == contacts_df["chromosome_2"]].copy()
    if cis_df.empty:
        raise ValueError("No cis-contacts found in the provided DataFrame.")

    cis_df["bin_1"] = cis_df["start_1"] // bin_size
    cis_df["bin_2"] = cis_df["start_2"] // bin_size

    chrom_list = cis_df["chromosome_1"].unique()
    all_tads = []

    # Process each chromosome independently
    for chrom in sorted(chrom_list):
        chrom_data = cis_df[cis_df["chromosome_1"] == chrom]
        bin_contacts = (
            chrom_data
            .groupby(["bin_1", "bin_2"])
            .size()
            .reset_index(name="contact_count")
        )

        max_bin_id = bin_contacts[["bin_1", "bin_2"]].max().max()
        N = max_bin_id + 1
        contact_matrix = np.zeros((N, N), dtype=float)
        for row in bin_contacts.itertuples(index=False):
            i = row.bin_1
            j = row.bin_2
            c = row.contact_count
            contact_matrix[i, j] += c
            contact_matrix[j, i] += c
        if imputation_involved:
            contact_matrix = imputation(
                contact_matrix,
                w=w,
                p=p
            )

        # Calculate directionality index and detect TAD boundaries
        di_vals = compute_directionality_index(contact_matrix, window=w)
        boundaries = detect_tad_boundaries(di_vals, threshold=boundary_threshold)
        tad_df_chrom = build_tad_df(chrom, boundaries, N, bin_size)
        all_tads.append(tad_df_chrom)
        plot_tads_for_chrom(
            contact_matrix=contact_matrix,
            boundaries=boundaries,
            chrom=chrom,
            out_prefix=out_prefix,
            show_plot=show_plot
        )

    tad_df = pd.concat(all_tads, ignore_index=True)

    stats_df = compute_tad_stats(tad_df)

    return {"stats_df": stats_df, "tad_df": tad_df}


def compute_cell_features(tad_stats: pd.DataFrame, tad_boundaries: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a feature vector for each chromosome based on TAD statistics.

    Parameters:
    -----------
    tad_stats : pd.DataFrame
        TAD statistics with columns:
        [chromosome, n_tads, mean_size_in_bins, min_size_in_bins, max_size_in_bins]

    tad_boundaries : pd.DataFrame
        TAD boundaries with columns:
        [chromosome, tad_id, start_bin, end_bin, start_bp, end_bp, size_in_bins]

    Returns:
    --------
    features_df : pd.DataFrame
        A feature vector for each chromosome with columns:
        [chromosome, n_tads, mean_size_in_bins, min_size_in_bins, max_size_in_bins,
         range_size_in_bins, median_size_in_bins, std_size_in_bins, tad_density,
         max_tad_ratio, min_tad_ratio]
    """
    feature_list = []

    for chrom, stats in tad_stats.groupby("chromosome"):
        genome_length_in_bins = tad_boundaries.loc[tad_boundaries["chromosome"] == chrom, "end_bin"].max() + 1
        tad_sizes = tad_boundaries.loc[tad_boundaries["chromosome"] == chrom, "size_in_bins"]

        # Extract relevant statistics from tad_stats
        n_tads = stats["n_tads"].values[0]
        mean_size = stats["mean_size_in_bins"].values[0]
        min_size = stats["min_size_in_bins"].values[0]
        max_size = stats["max_size_in_bins"].values[0]

        # Compute additional features
        range_size = max_size - min_size
        median_size = tad_sizes.median()
        std_size = tad_sizes.std()
        tad_density = n_tads / genome_length_in_bins
        max_tad_ratio = max_size / genome_length_in_bins
        min_tad_ratio = min_size / genome_length_in_bins

        feature_list.append({
            "chromosome": chrom,
            "n_tads": n_tads,
            "mean_size_in_bins": mean_size,
            "min_size_in_bins": min_size,
            "max_size_in_bins": max_size,
            "range_size_in_bins": range_size,
            "median_size_in_bins": median_size,
            "std_size_in_bins": std_size,
            "tad_density": tad_density,
            "max_tad_ratio": max_tad_ratio,
            "min_tad_ratio": min_tad_ratio,
        })

    features_df = pd.DataFrame(feature_list)
    return features_df


def compute_tad_features(
    contacts_df: pd.DataFrame,
    bin_size: int = 1_000_000,
    w: int = 10,
    p: float = 0.85,
    imputation_involved: bool = False,
    boundary_threshold: float = 0.3,
    out_prefix: str = None,
    show_plot: bool = False
):
    """
    Orchestrates TAD detection and feature computation for each chromosome.

    Parameters:
    -----------
    contacts_df : pd.DataFrame
        Input DataFrame containing contact information (chromosome_1, start_1, etc.).
    bin_size : int, optional
        Bin size in base pairs; defaults to 1,000,000.
    w : int, optional
        Window size for TAD boundary detection (and possibly imputation), default 10.
    p : float, optional
        Probability parameter for the random walk with restart (imputation), default 0.85.
    threshold_percentile : int, optional
        Threshold percentile for binarization in imputation, default 91.
    imputation_involved : bool, optional
        Whether to run imputation on the contact matrix, default False.
    boundary_threshold : float, optional
        Threshold for TAD boundary detection based on directionality index, default 0.3.
    out_prefix : str, optional
        Prefix to save plots to files; if None, no files are saved.
    show_plot : bool, optional
        Whether to display the plot (True) or not (False), default False.

    Returns:
    --------
    dictionary
        A dictionary containing the average number of TADs, average mean TAD size in bins,
        and average TAD density across all chromosomes.
    """
    tad_results = calculate_cis_tads(
        contacts_df,
        bin_size=bin_size,
        w=w,
        p=p,
        imputation_involved=imputation_involved,
        boundary_threshold=boundary_threshold,
        out_prefix=out_prefix,
        show_plot=show_plot
    )

    # The returned dictionary from calculate_cis_tads 
    # has keys "stats_df" (TAD stats) and "tad_df" (TAD boundaries)
    tad_stats = tad_results["stats_df"]
    tad_boundaries = tad_results["tad_df"]

    features_df = compute_cell_features(tad_stats, tad_boundaries)

    n_tads_mean = gmean(features_df['n_tads'])
    mean_bin_size = gmean(features_df['mean_size_in_bins'])
    tad_density_mean = gmean(features_df['tad_density'])

    return {
        "tad_n_tads_mean": n_tads_mean,
        "tad_mean_bin_size": mean_bin_size,
        "tad_density_mean": tad_density_mean
    }