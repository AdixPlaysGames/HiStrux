import pandas as pd
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
from .imputation import imputation
from .visualization import visualize

def compute_ab_compartments(
    contacts_df: pd.DataFrame,
    bin_size: int = 1_000_000, 
    w: int = 4, 
    p: float = 0.85, 
    imputation_involved: bool = True,
    plot: bool = False
) -> pd.DataFrame:
    """
    Computes PC1 (the first principal component) for each bin in the genome
    to identify A/B compartments using PCA.

    Parameters:
    -----------
    contacts_df : pd.DataFrame
        Input DataFrame with the following columns:
          - chromosome_1
          - start_1
          - end_1
          - chromosome_2
          - start_2
          - end_2

    bin_size : int, optional
        The bin size in base pairs; defaults to 1,000,000 bp (1 Mb).
    w : int, optional
        Parameter for the imputation function (if used). Defaults to 4.
    p : float, optional
        Another parameter for the imputation function (if used). Defaults to 0.85.
    imputation_involved : bool, optional
        Whether to apply the imputation step on the contact matrix. Defaults to False.
    plot : bool, optional
        Display each chromosome for better undrestanding.
    Returns:
    --------
    result_df : pd.DataFrame
        A DataFrame with the following columns:
          - chromosome
          - bin_start
          - bin_end
          - PC1
          - compartment_label  (e.g., 'A' or 'B')
        The rows are sorted in ascending order by chromosome and bin.
    """

    cis_df = contacts_df[contacts_df["chromosome_1"] == contacts_df["chromosome_2"]].copy()
    if len(cis_df) == 0:
        raise ValueError("No cis-contacts found in the input DataFrame.")

    def get_bin_id(start, bin_size):
        return start // bin_size

    cis_df["bin_1"] = cis_df["start_1"].apply(lambda x: get_bin_id(x, bin_size))
    cis_df["bin_2"] = cis_df["start_2"].apply(lambda x: get_bin_id(x, bin_size))

    chrom_list = cis_df["chromosome_1"].unique()
    results = []

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

        if imputation_involved is True:
            contact_matrix = imputation(
                contact_matrix, 
                w=w, 
                p=p, 
                # threshold_percentile=threshold_percentile
                # can be changed anytime but preferred 
            )
        
        if plot is True:
          visualize(contact_matrix, title=chrom)

        with np.errstate(invalid='ignore'):
            corr_matrix = np.corrcoef(contact_matrix)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Perform PCA on the correlation matrix.
        pca = PCA(n_components=1)
        X = corr_matrix
        pca.fit(X)

        # PC1 scores for each bin (size N).
        pc1_coords = pca.transform(X)[:, 0]
        data_out = []
        for bin_id in range(N):
            start_bp = bin_id * bin_size
            end_bp = (bin_id + 1) * bin_size - 1
            val = pc1_coords[bin_id]
            label = "A" if val >= 0 else "B"
            data_out.append((chrom, bin_id, start_bp, end_bp, val, label))

        chrom_df = pd.DataFrame(
            data_out,
            columns=["chromosome", "bin_id", "bin_start", "bin_end", "PC1", "compartment_label"]
        )
        results.append(chrom_df)

    result_df = pd.concat(results, ignore_index=True)
    result_df.sort_values(by=["chromosome", "bin_start"], inplace=True)

    return result_df



def compute_ab_stats(result_df: pd.DataFrame,
                     contacts_df: pd.DataFrame, 
                     bin_size: int = 1_000_000):
    """
    Function that calculates the 'cis AB fraction':
    the proportion of A-A, B-B, and A-B contacts in cis (for the entire genome or per chromosome).

    Parameters:
    -----------
    result_df : pd.DataFrame
        The result of the `compute_ab_compartments` function, with columns:
        [chromosome, bin_id, bin_start, bin_end, PC1, compartment_label]
    contacts_df : pd.DataFrame
        The original contacts DataFrame (where chromosome_1 == chromosome_2).
    bin_size : int
        Bin size in base pairs.
    Returns:
    --------
    ab_stats : pd.DataFrame
        A table counting the number of A-A, B-B, and A-B contacts (and their fractions).
    """
    cis_df = contacts_df[contacts_df["chromosome_1"] == contacts_df["chromosome_2"]].copy()
    def get_bin_id(start):
        return start // bin_size

    cis_df["bin_1"] = cis_df["start_1"].apply(get_bin_id)
    cis_df["bin_2"] = cis_df["start_2"].apply(get_bin_id)

    compartment_map = {}
    for row in result_df.itertuples():
        compartment_map[(row.chromosome, row.bin_id)] = row.compartment_label

    stats = {"AA": 0, "BB": 0, "AB": 0}
    for row in cis_df.itertuples():
        chrom = row.chromosome_1
        b1 = row.bin_1
        b2 = row.bin_2
        c1 = compartment_map.get((chrom, b1), None)
        c2 = compartment_map.get((chrom, b2), None)
        if c1 is not None and c2 is not None:
            if c1 == "A" and c2 == "A":
                stats["AA"] += 1
            elif c1 == "B" and c2 == "B":
                stats["BB"] += 1
            else:
                stats["AB"] += 1

    total = stats["AA"] + stats["BB"] + stats["AB"]
    if total == 0:
        return pd.DataFrame({"contact_type": [], "count": [], "fraction": []})

    contact_types = []
    for k in ["AA", "BB", "AB"]:
        contact_types.append(
            {
                "contact_type": k,
                "count": stats[k],
                "fraction": stats[k] / total
            }
        )
    ab_stats = pd.DataFrame(contact_types)

    return ab_stats

def calculate_cis_ab_comp(
    contacts_df: pd.DataFrame,
    bin_size: int = 1_000_000,
    w: int = 4,
    p: float = 0.85,
    imputation_involved=False,
    plot: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates A/B compartments and then computes A-B contact statistics.

    Parameters:
    -----------
    contacts_df : pd.DataFrame
        Input DataFrame with contact information (including columns chromosome_1, chromosome_2, etc.).
    bin_size : int, optional
        The bin size in base pairs, default 1,000,000 bp.
    w : int, optional
        Parameter passed to the imputation function, default 5.
    p : float, optional
        Parameter passed to the imputation function, default 0.85.
    threshold_percentile : int, optional
        Percentile threshold for the imputation function, default 90.
    imputation_involved : bool, optional
        Whether to run imputation on the contact matrix, default False.

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        - A/B compartment statistics DataFrame.
    """

    # Compute A/B compartments
    compartments_df = compute_ab_compartments(
        contacts_df=contacts_df,
        bin_size=bin_size,
        w=w,
        p=p,
        imputation_involved=imputation_involved,
        plot=plot
    )
    # Compute A-B statistics
    ab_stats_df = compute_ab_stats(
        result_df=compartments_df,
        contacts_df=contacts_df,
        bin_size=bin_size
    )

    return ab_stats_df