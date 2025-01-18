import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema # type: ignore
import pandas as pd
from .imputation import imputation
from .process import process
from .visualization import visualize

def compute_insulation_scores(cell: np.ndarray, 
                              scale: int, 
                              apply_smoothing: bool = True, 
                              plot: bool=False) -> np.ndarray:
    """
    Calculates the insulation score with an optional smoothing feature.

    Parameters:
    - cell: np.ndarray - contact matrix (2D array).
    - scale: int - size of the neighborhood (number of bins) around each bin.
    - apply_smoothing: bool - True if smoothing (local normalization) is to be applied.
    - plot: bool - True if the insulation scores should be visualized as a plot.

    Returns:
    - insulation_scores: np.ndarray - insulation scores after normalization or Z-score standardization.
    """
    N = cell.shape[0]  # Size of the contact matrix (number of bins).
    ins_scores = np.zeros(N, dtype=float)  # Initialize insulation scores array.

    # Calculate insulation scores for each bin.
    for b in range(N):
        # Define the neighborhood boundaries for the current bin.
        left_start = max(b - scale, 0)
        left_end = b
        right_start = min(b + 1, N - 1)
        right_end = min(b + scale, N - 1)

        # Extract the submatrix (cross-block) and sum its values.
        if left_end >= left_start and right_end >= right_start:
            cross_block = cell[left_start:left_end+1, right_start:right_end+1]
            ins_scores[b] = np.nansum(cross_block)  # Sum all values in the submatrix.
        else:
            ins_scores[b] = 0.0  # If boundaries are invalid, set the score to 0.

    # Replace NaN values in the insulation scores with 0.
    ins_scores = np.nan_to_num(ins_scores, nan=0.0)

    # Logarithmic transformation to stabilize variance and handle zeros.
    epsilon = 1e-6  # Small constant to avoid log(0).
    log_scores = np.log(ins_scores + epsilon)

    # If smoothing is enabled, apply local normalization.
    if apply_smoothing:
        normalized_scores = np.zeros_like(log_scores)  # Initialize normalized scores array.
        window = scale  # Define the size of the local window for normalization.
        for i in range(N):
            # Define the local window boundaries.
            local_start = max(0, i - window)
            local_end = min(N, i + window + 1)
            local_scores = log_scores[local_start:local_end]

            # Compute the min and max of the local window.
            local_min = np.min(local_scores)
            local_max = np.max(local_scores)

            # Normalize the score based on the local range.
            if local_max - local_min > epsilon:
                normalized_scores[i] = (log_scores[i] - local_min) / (local_max - local_min)
            else:
                normalized_scores[i] = 0.5  # If no variation in the window, assign a default value.

        # Center the normalized scores around zero.
        final_scores = normalized_scores - np.mean(normalized_scores)

    else:
        # Standard Z-score normalization (global).
        mean_ = np.mean(log_scores)  # Compute the global mean.
        std_ = np.std(log_scores)  # Compute the global standard deviation.

        # If the standard deviation is very small, use raw log_scores.
        if std_ < 1e-12:
            z_scores = log_scores
        else:
            z_scores = (log_scores - mean_) / std_

        final_scores = z_scores  # Assign Z-scored values as the final scores.

    # If the plot flag is True, visualize the insulation scores.
    if plot == True:
        plt.figure(figsize=(12, 6))
        plt.plot(final_scores, label='Insulation Score')  # Plot the final scores.
        plt.xlabel('Bin Index')  # Label the x-axis.
        plt.ylabel('Insulation Score')  # Label the y-axis.
        plt.title('Insulation Score Across Chromosome')  # Title of the plot.
        plt.legend()  # Add a legend to the plot.
        plt.show()  # Display the plot.

    return final_scores  # Return the final insulation scores.


def compute_insulation_features(ins_scores: np.ndarray, plot: bool = True, chrom: str=None) -> dict:
    """
    Analyzes local minima in insulation scores, groups them between local maxima, 
    computes metrics (mean, sum, standard deviation), and optionally plots the results.

    Parameters:
    - ins_scores: np.ndarray - insulation score vector.
    - plot: bool - whether to plot the results (default: True).

    Returns:
    - result: dict - dictionary with groups, group averages, mean, sum, and standard deviation.
    """
    # Find local minima
    local_min_indices = np.array([
        i for i in range(1, len(ins_scores) - 1)
        if ins_scores[i] <= ins_scores[i - 1] and ins_scores[i] <= ins_scores[i + 1]
    ])
    
    # Find local maxima
    local_max_indices = np.array([
        i for i in range(1, len(ins_scores) - 1)
        if ins_scores[i] >= ins_scores[i - 1] and ins_scores[i] >= ins_scores[i + 1]
    ])
    
    # Add start and end as "artificial" maxima
    local_max_indices = np.sort(np.concatenate(([0], local_max_indices, [len(ins_scores) - 1])))

    # Group minima between maxima
    groups = []
    for i in range(len(local_max_indices) - 1):
        start = local_max_indices[i]
        end = local_max_indices[i + 1]
        group = [idx for idx in local_min_indices if start < idx < end]
        groups.append(group)

    # Calculate average positions and values for each group
    group_averages = []
    for group in groups:
        if len(group) > 0:
            avg_position = np.mean(group)
            avg_value = np.mean(ins_scores[group])
            group_averages.append((avg_position, avg_value))

    # Exclude the first and last group for metrics
    filtered_group_averages = group_averages[1:-1]
    avg_values = [value for _, value in filtered_group_averages]

    # Compute metrics
    mean_value = np.mean(avg_values) if avg_values else 0.0
    sum_value = np.sum(avg_values) if avg_values else 0.0
    std_deviation = np.std(avg_values) if avg_values else 0.0

    minima_counts_new = []

    for i in range(len(local_max_indices) - 1):
        start = local_max_indices[i]
        end = local_max_indices[i + 1]

        # Count minima between current maximum and the next one
        count = sum((start < idx < end) for idx in local_min_indices)
        minima_counts_new.append(count)


    peak_distances = np.diff(local_max_indices)


    # Create result dictionary
    result = {
        "groups": groups,
        "group_averages": group_averages,
        "mean_value": mean_value,
        "sum_value": sum_value,
        "std_deviation": std_deviation,
        "std_minima_counts": np.std(minima_counts_new),
        "mean_peak_distances": np.mean(peak_distances.tolist())
    }

    if plot:
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(ins_scores, label='Insulation Score', linewidth=2)
        plt.xlabel('Bin Index', fontsize=14)
        plt.ylabel('Insulation Score', fontsize=14)
        plt.title(f'Insulation Score with Grouped Minima Averages {chrom}', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)

        # Add vertical lines for local minima
        for idx in local_min_indices:
            plt.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
        
        # Add vertical lines for local maxima
        for idx in local_max_indices:
            plt.axvline(x=idx, color='blue', linestyle=':', alpha=0.5)

        # Plot group averages
        for avg_position, avg_value in group_averages:
            plt.scatter(avg_position, avg_value, color='green', s=100, label='Group Average')

        plt.show()

    return result


def compute_ins_features_for_each_chr(contacts_df: pd.DataFrame, 
                     bin_size: int=400_000, 
                     plot: bool = False,
                     plot_insulation = False, 
                     imputation_involved: bool = True,
                     w: int = 3,
                     p: float = 0.85,
                     scale: int = 15) -> list:
    """
    Compute insulation features for all chromosomes and return results as a list of dictionaries.

    Parameters:
    - contacts_df: pd.DataFrame - input dataframe with Hi-C contacts.
    - bin_size: int - size of bins for grouping.
    - plot: bool - whether to plot results (default: False).
    - imputation_involved: bool - whether to apply imputation (default: False).

    Returns:
    - results: list - list of dictionaries with insulation features for each chromosome.
    """
    # Filter for cis-contacts
    cis_df = contacts_df[contacts_df["chromosome_1"] == contacts_df["chromosome_2"]].copy()

    chrom_list = cis_df["chromosome_1"].unique()
    results = []

    for chrom in sorted(chrom_list):
        chrom_data = cis_df[cis_df["chromosome_1"] == chrom]

        contact_matrix = process(chrom_data, bin_size = bin_size, 
                                 selected_chromosomes=[chrom], trans_interactions=False, substring=None)

        # Apply imputation if required
        if imputation_involved:
            contact_matrix = imputation(contact_matrix, w=w, p=p)
        
        if plot is True:
          visualize(contact_matrix, title=chrom)

        # Compute insulation scores and features
        ins_scores = compute_insulation_scores(contact_matrix, scale=scale, apply_smoothing=False, plot=False)
        features = compute_insulation_features(ins_scores, plot=plot_insulation, chrom=chrom)
        result = [chrom]
        result += [[key, value] for key, value in features.items() if key in {"std_deviation"}] #, "mean_value", "sum_value", "mean_peak_distances", "std_minima_counts"}]
        results.append(result)

    return results