import pandas as pd
from HiStrux.eXtract.process import process
from HiStrux.eXtract.cdd import compute_cdd

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

chromosome_lengths = [('chr1', 195471971), ('chr2', 182113224), ('chr3', 160039680), ('chr4', 156508116), 
                      ('chr5', 151834684), ('chr6', 149736546), ('chr7', 145441459), ('chr8', 129401213), 
                      ('chr9', 124595110), ('chr10', 130694993), ('chr11', 122082543), ('chr12', 120129022),
                      ('chr13', 120421639), ('chr14', 124902244), ('chr15', 104043685), ('chr16', 98207768), 
                      ('chr17', 94987271), ('chr18', 90702639), ('chr19', 61431566), ('chrX', 171031299)]

cell_id = 'SCG0088_TTGTGTGCACGGTACT-1'
path = "C:/Users/zareb/OneDrive/Desktop/Studies/Inżynierka/CIRCLET/CIRCLET_code/CIRCLET/patski.S_5.two.bedpe"
cell_df = pd.read_csv(path, sep="\t", names=columns, comment='#')
cell_matrix = process(cell_df, chromosome_lengths=chromosome_lengths, bin_size=500_000, cell_id=cell_id)

print(compute_cdd(cell=cell_matrix, plot=True))