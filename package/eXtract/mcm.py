import numpy as np

def compute_mcm(hic_matrix: np.ndarray,
                bin_size: int=1_000_000,
                near_threshold: float=2.0,
                mid_threshold: float=5.0):
    """
    Computes a multi-class metric (MCM) vector, which reflects the proportions of Hi-C contacts
    classified into three distance categories: 'near', 'mid', and 'far'.

    Parameters:
    -----------
    hic_matrix : np.ndarray
        A square Hi-C contact matrix (N x N), where hic_matrix[i, j] is the contact count
        between bin i and j. Assumes 1 Mb bins (or a specified bin_size) and a symmetric matrix.
    bin_size : int, optional
        The size of each bin in base pairs (bp). Default is 1,000,000 (1 Mb).
    near_threshold : float, optional
        A distance threshold in megabases (Mb) below which contacts are considered "near".
        Default is 2.0 Mb.
    mid_threshold : float, optional
        A distance threshold in megabases (Mb) above which contacts are considered "far".
        Contacts between near_threshold and mid_threshold are "mid". Default is 5.0 Mb.

    Returns:
    --------
    mcm_dict :
        A dictionary of three values [near_ratio, mid_ratio, far_ratio], representing the fraction
        of contacts in each distance category.
    """

    hic_matrix = (hic_matrix > 0).astype(int)

    def compute_distance_classes(hic_matrix, bin_size=1_000_000,
                                near_threshold=2.0,
                                mid_threshold=5.0):

        if not np.allclose(hic_matrix, hic_matrix.T):
            hic_matrix = (hic_matrix + hic_matrix.T) / 2.0

        N = hic_matrix.shape[0]
        i_coords, j_coords = np.indices((N, N))
        dist = np.abs(i_coords - j_coords) * (bin_size / 1_000_000.0)  # Distance in Mb

        mask = (i_coords < j_coords)
        contacts = hic_matrix[mask]
        distances = dist[mask]

        total_contacts = np.sum(contacts)
        if total_contacts == 0:
            return [0.0, 0.0, 0.0]

        # Create masks for each distance category
        near_mask = distances < near_threshold
        far_mask = distances > mid_threshold
        mid_mask = ~near_mask & ~far_mask

        near_contacts = np.sum(contacts[near_mask])
        mid_contacts = np.sum(contacts[mid_mask])
        far_contacts = np.sum(contacts[far_mask])

        # Calculate proportions
        near_ratio = near_contacts / total_contacts if total_contacts > 0 else 0.0
        mid_ratio = mid_contacts / total_contacts if total_contacts > 0 else 0.0
        far_ratio = far_contacts / total_contacts if total_contacts > 0 else 0.0

        return {
            'mcm_near_ratio': near_ratio,
            'mcm_mid_ratio': mid_ratio,
            'mcm_far_ratio': far_ratio
        }

    # Compute the distance class ratios
    mcm_dict = compute_distance_classes(hic_matrix, bin_size=bin_size,
                                          near_threshold=near_threshold,
                                          mid_threshold=mid_threshold)

    # This function is created based on CIRCLET tool. Basic data that contains
    # basic values such as chromosome id, cell id, contact length and mapping quality
    # doesn't support other calculations. However it can be expanded making MCM
    # much more powerful.
    return mcm_dict