import pandas as pd
from package.eXtract.tad import calculate_cis_tads, compute_tad_features

columns = [
    "chromosome_1",  # First chromosome
    "start_1",       # Start of the first fragment
    "end_1",         # End of the first fragment
    "chromosome_2",  # Second chromosome
    "start_2",       # Start of the second fragment
    "end_2",         # End of the second fragment
    "cell_id",       # Cell identifier
    "read_id",       # Read identifier
    "mapping_quality",  # Mapping quality
    "strand_1",      # DNA strand for the first fragment
    "strand_2"       # DNA strand for the second fragment
]


path = "C:/Users/zareb/OneDrive/Desktop/Studies/Inżynierka/CIRCLET/CIRCLET_code/CIRCLET/patski.S_5.two.bedpe"
cell_df = pd.read_csv(path, sep="\t", names=columns, comment='#')
cell_df = cell_df[cell_df['cell_id'] == 'SCG0088_TTGTGTGCACGGTACT-1']

print(calculate_cis_tads(
    cell_df,
    bin_size=500_000,
    w=10,
    p=0.90,
    threshold_percentile=85,
    imputation_involved=True,
    boundary_threshold=0.19,
    show_plot=True
))


print(compute_tad_features(
    cell_df,
    bin_size=500_000,
    w=10,
    p=0.90,
    threshold_percentile=85,
    imputation_involved=True,
    boundary_threshold=0.19,
    show_plot=False
))