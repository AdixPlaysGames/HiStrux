import numpy as np
import pandas as pd

def process(
    cells: pd.DataFrame,
    cell_id: str = None,
    chromosome_lengths: list[tuple[str, int]] = None,
    bin_size: int = 1_000_000,
    selected_chromosomes: list[str] = None,
    trans_interactions: bool = True,
    mapping_quality_involved: bool = False,
    substring = 2
) -> np.ndarray:
    """
    Processes a single-cell Hi-C dataset to produce a contact matrix.

    Parameters:
    - cells: pd.DataFrame
        The input dataframe containing Hi-C data. Required columns include:
        ['cell_id', 'chromosome_1', 'start_1', 'chromosome_2', 'start_2', 'mapping_quality'].
        Raises a ValueError if the input is not a pandas DataFrame or if required columns are missing.
    - cell_id: str
        Identifier for the specific cell to process. If None, the first cell_id in the dataframe will be used.
    - chromosome_lengths: list of tuples
        A list of tuples containing chromosome names and their lengths.
        Example: [('chr1', 195471971), ('chr2', 182113224), ...].
    - bin_size: int
        The size of each bin in base pairs. Default is 1,000,000.
    - selected_chromosomes: list of str
        A list of chromosome names to include in the analysis. If None, all chromosomes in chromosome_lengths are used.
    - trans_interactions: bool
        If True, include inter-chromosomal interactions. If False, only intra-chromosomal interactions are considered.
    - mapping_quality_involved: bool
        If True, the contact matrix will sum the mapping qualities for each bin. If False, it will count the number of interactions per bin.
    - substring : 
        For chromosome name it removes the last (substring) characters for name reduction.
        Example: chromosome_1 = chr1-P, then for substring = 2 our chromosome_1 = chr1.
        IMPORTANT, if You want to stay with name you need to set None.

    Returns:
    - contact_matrix: np.ndarray
        A symmetric contact matrix where rows and columns represent genomic bins.

    Raises:
    - ValueError: If the input 'cells' is not a DataFrame or if required columns are missing.
    """

    # Check if 'cells' is indeed a pandas DataFrame
    if not isinstance(cells, pd.DataFrame):
        raise ValueError("Input 'cells' must be a pandas DataFrame.")

    # Ensure all required columns are present in the dataframe
    required_columns = {'cell_id', 'chromosome_1', 'start_1', 'chromosome_2', 'start_2', 'mapping_quality'}
    if not required_columns.issubset(cells.columns):
        raise ValueError(f"Input DataFrame must contain the following columns: {required_columns}")
  
    # If no cell_id is provided, use the first one found in the DataFrame
    if cell_id is None:
      cells = cells.reset_index(drop=True)
      cell_id = cells['cell_id'][0]
    
    # Create a copy of the relevant data for the specified cell
    cell = cells[cells['cell_id'] == cell_id].copy()

    # If substring is not None, trim the last 'substring' characters from chromosome names
    if substring is not None:
        cell['chromosome_1'] = cell['chromosome_1'].str[:-substring]
        cell['chromosome_2'] = cell['chromosome_2'].str[:-substring]

    # Filter for intra-chromosomal interactions if trans_interactions is False
    if trans_interactions is False:
        cell = cell[cell['chromosome_1'] == cell['chromosome_2']]

    # Assign default mouse chromosome lengths if none are provided
    # These values are basic, provided from the website
    if chromosome_lengths is None:
        chromosome_lengths = [('chr1', 195471971), ('chr2', 182113224), ('chr3', 160039680), ('chr4', 156508116), 
                              ('chr5', 151834684), ('chr6', 149736546), ('chr7', 145441459), ('chr8', 129401213), 
                              ('chr9', 124595110), ('chr10', 130694993), ('chr11', 122082543), ('chr12', 120129022),
                              ('chr13', 120421639), ('chr14', 124902244), ('chr15', 104043685), ('chr16', 98207768), 
                              ('chr17', 94987271), ('chr18', 90702639), ('chr19', 61431566), ('chrX', 171031299)]
        
    # If specific chromosomes are selected, filter the chromosome_lengths accordingly
    if selected_chromosomes is None:
        selected_chromosomes = [chrom[0] for chrom in chromosome_lengths]
    else:
        chromosome_lengths = [chrom for chrom in chromosome_lengths if chrom[0] in selected_chromosomes]
    
    # Calculate cumulative offsets for each chromosome so bins can be positioned along the genome
    chromosome_offsets = np.cumsum([0] + [np.int64(length) for _, length in chromosome_lengths], dtype=np.int64)
    chromosome_map = {chrom: offset for (chrom, _), offset in zip(chromosome_lengths, chromosome_offsets)}

    cell['position_1'] = cell['start_1'] + cell['chromosome_1'].map(chromosome_map)
    cell['position_2'] = cell['start_2'] + cell['chromosome_2'].map(chromosome_map)

    # Determine bins based on bin_size
    cell['bin1'] = cell['position_1'] // bin_size
    cell['bin2'] = cell['position_2'] // bin_size

    # Decide how the contact information should be aggregated:
    # If mapping_quality_involved is False, we are summing the "contact_weight" (which is contact size * mapping quality).
    # Otherwise, we simply count interactions (grouped size).
    if not mapping_quality_involved:
        # Calculate the average contact size (mean of the length of read fragments)
        cell['contact_size'] = ((cell['end_1'] - cell['start_1']).abs().mean() + 
                                (cell['end_2'] - cell['start_2']).abs().mean()) / 2
        
        # Define the weight of the contact as the product of contact size and mapping quality
        cell['contact_weight'] = (cell['contact_size'] * cell['mapping_quality']).round().astype(int)

        grouped = cell.groupby(['bin1', 'bin2'], as_index=False)['contact_weight'].sum()
        grouped_array = grouped.to_numpy()
    else:
        grouped = cell.groupby(['bin1', 'bin2'], as_index=False).size()
        grouped_array = grouped.to_numpy()

    # Prepare the contact matrix of size (number of bins x number of bins)
    total_genome_length = sum(length for _, length in chromosome_lengths)
    num_bins = total_genome_length // bin_size
    contact_matrix = np.zeros((num_bins, num_bins), dtype=int)

    # Ensure bin indices do not exceed the contact matrix dimensions as also determine and report the number of bins that were excluded
    max_bin = num_bins - 1
    mask = (grouped_array[:, 0].astype(int) <= max_bin) & (grouped_array[:, 1].astype(int) <= max_bin)
    cut_bins = len(grouped_array) - np.sum(mask)
    grouped_array = grouped_array[mask]

    if cut_bins > 0:
        print(f"Cut {cut_bins} bins that exceeded the matrix range due to provided chromosome lengths in the input. If more than one bin was cut, please check the correctness of the chromosome lengths.")
    
    contact_matrix[grouped_array[:, 0].astype(int), grouped_array[:, 1].astype(int)] = grouped_array[:, 2].astype(int)
    contact_matrix += contact_matrix.T - np.diag(contact_matrix.diagonal())

    return contact_matrix