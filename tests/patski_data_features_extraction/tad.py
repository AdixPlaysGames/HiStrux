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
cell_df = cell_df[cell_df['cell_id'] == 'SCG0089_TCATGCCTCCCGTTAC-1']
cell_df['chromosome_1'] = cell_df['chromosome_1'].str[:-2]
cell_df['chromosome_2'] = cell_df['chromosome_2'].str[:-2]



tad = compute_tad_features(
    cell_df,
    bin_size=600_000,
    w=3,
    p=0.85,
    imputation_involved=True,
    boundary_threshold=0.05,
    show_plot=False
)

vector = []
values = []
vector += [value for key, value in tad.items()]
values += [key for key, value in tad.items()]
# 'tad_n_tads_mean', 'tad_mean_bin_size', 'tad_density_mean'

print(vector)
print(values)