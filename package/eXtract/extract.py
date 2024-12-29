import pandas as pd
from .process import process
from .primary import compute_basic_metrics
from .compartments import compute_ab_stats, calculate_cis_ab_comp
from .cdd import compute_cdd


# process
def eXtract(cell_df: pd.DataFrame,
            cell_id: str = None,
            chromosome_lengths: list[tuple[str, int]] = None,
            bin_size: int = 1_000_000,
            selected_chromosomes: list[str] = None,
            trans_interactions: bool = True,
            mapping_quality_involved: bool = False,
            substring = 2,

            # compartments
            w: int = 11,
            p: float = 0.92,
            threshold_percentile: int = 85,
            imputation_involved: bool = False,
            boundry_threshold: int = 0.5

            ) -> dict:
    
    if cell_id is None:
        cell_id = cell_df['cell_id'][0]

    cell = cell_df[cell_df['cell_id'] == cell_id].copy()

    cell_matrix = process(cell, cell_id=cell_id, chromosome_lengths=chromosome_lengths, bin_size=bin_size,
                      selected_chromosomes=selected_chromosomes, trans_interactions=trans_interactions, 
                      mapping_quality_involved=mapping_quality_involved, substring=substring)
    
    if substring is not None:
        cell['chromosome_1'] = cell['chromosome_1'].str[:-substring]
        cell['chromosome_2'] = cell['chromosome_2'].str[:-substring]

    def partition_and_calculate_means(data, partitions=7):
        partition_size = len(data) // partitions
        partitioned_data = [data[i * partition_size:(i + 1) * partition_size] for i in range((partitions - 1))]
        partitioned_data.append(data[(partitions - 1) * partition_size:])  # Ostatnia partycja może być większa
        means = [sum(partition) / len(partition) if partition else 0 for partition in partitioned_data]
        return means
    cdd = partition_and_calculate_means(compute_cdd(cell_matrix, bin_size=bin_size)['probs_array'], partitions=7)


    compartments = calculate_cis_ab_comp(
        contacts_df=cell, bin_size=bin_size, w=w, p=p, threshold_percentile=threshold_percentile, 
        imputation_involved=imputation_involved, plot=False
    )
    
    return compartments