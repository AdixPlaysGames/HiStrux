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
    substring: int = 2
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
    - substring : int
        For chromosome name it removes the last (substring) characters for name reduction.
        Example: chromosome_1 = chr1-P, then for substring = 2 our chromosome_1 = chr1.

    Returns:
    - contact_matrix: np.ndarray
        A symmetric contact matrix where rows and columns represent genomic bins.

    Raises:
    - ValueError: If the input 'cells' is not a DataFrame or if required columns are missing.
    """

    # Validate input format
    if not isinstance(cells, pd.DataFrame):
        raise ValueError("Input 'cells' must be a pandas DataFrame.")

    required_columns = {'cell_id', 'chromosome_1', 'start_1', 'chromosome_2', 'start_2', 'mapping_quality'}
    if not required_columns.issubset(cells.columns):
        raise ValueError(f"Input DataFrame must contain the following columns: {required_columns}")
  
    if cell_id is None:
      cell_id = cells['cell_id'][0]
    
    cell = cells[cells['cell_id'] == cell_id].copy()

    if trans_interactions is False:
        cell = cell[cell['chromosome_1'] == cell['chromosome_2']]
    
    cell['chromosome_1'] = cell['chromosome_1'].str[:-substring]
    cell['chromosome_2'] = cell['chromosome_2'].str[:-substring]

    if chromosome_lengths is None:
        chromosome_lengths = [('chr1', 195471971), ('chr2', 182113224), ('chr3', 160039680), ('chr4', 156508116), 
                              ('chr5', 151834684), ('chr6', 149736546), ('chr7', 145441459), ('chr8', 129401213), 
                              ('chr9', 124595110), ('chr10', 130694993), ('chr11', 122082543), ('chr12', 120129022),
                              ('chr13', 120421639), ('chr14', 124902244), ('chr15', 104043685), ('chr16', 98207768), 
                              ('chr17', 94987271), ('chr18', 90702639), ('chr19', 61431566), ('chrX', 171031299)]
        
    if selected_chromosomes is None:
        selected_chromosomes = [chrom[0] for chrom in chromosome_lengths]
    else:
        chromosome_lengths = [chrom for chrom in chromosome_lengths if chrom[0] in selected_chromosomes]
    
    chromosome_offsets = np.cumsum([0] + [np.int64(length) for _, length in chromosome_lengths], dtype=np.int64)
    chromosome_map = {chrom: offset for (chrom, _), offset in zip(chromosome_lengths, chromosome_offsets)}


    cell['position_1'] = cell['start_1'] + cell['chromosome_1'].map(chromosome_map)
    cell['position_2'] = cell['start_2'] + cell['chromosome_2'].map(chromosome_map)

    # Przypisz biny
    cell['bin1'] = cell['position_1'] // bin_size
    cell['bin2'] = cell['position_2'] // bin_size

    if not mapping_quality_involved:
        # Obliczanie średniego rozmiaru kontaktu
        cell['contact_size'] = ((cell['end_1'] - cell['start_1']).abs().mean() + 
                                (cell['end_2'] - cell['start_2']).abs().mean()) / 2
        
        # Wyznaczenie wagi kontaktu
        cell['contact_weight'] = (cell['contact_size'] * cell['mapping_quality']).round().astype(int)

        # Grupowanie i sumowanie wag kontaktów
        grouped = cell.groupby(['bin1', 'bin2'], as_index=False)['contact_weight'].sum()
        grouped_array = grouped.to_numpy()
    else:
        # Grupowanie i liczenie liczby kontaktów
        grouped = cell.groupby(['bin1', 'bin2'], as_index=False).size()
        grouped_array = grouped.to_numpy()

    # Przygotuj macierz kontaktów
    total_genome_length = sum(length for _, length in chromosome_lengths)
    num_bins = total_genome_length // bin_size
    contact_matrix = np.zeros((num_bins, num_bins), dtype=int)

    # Przypisz liczby interakcji do macierzy
    contact_matrix[grouped_array[:, 0].astype(int), grouped_array[:, 1].astype(int)] = grouped_array[:, 2].astype(int)
    contact_matrix += contact_matrix.T - np.diag(contact_matrix.diagonal())

    return contact_matrix