import numpy as np
import matplotlib.pyplot as plt

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


def compute_insulation_features(cell: np.ndarray, scale: int=15) -> dict:
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
        "ins_vector": ins_scores
    }