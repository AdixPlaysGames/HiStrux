import numpy as np
import pandas as pd
import cooler
from typing import Optional, Union
import random
import h5py

# data selection definitions
def load_cells_names(population_dir: str, 
                     cells_num: Optional[int] = 10,
                     cells_names: Optional[list[str]] = None) -> list[str]:
    """
    Reads required number of cell names from scool file taking first cells. If user knows the names of cells he want to work with this, function can be used as a check whether those cells are actually in the file.

    Parameters
    ----------
    population_dir : str
        Scool file directory.
    num : optional[int] = 10
        Number of cell's names to be read.
    cells_names : Optional[list[str]] = None
        Name of cell's names to be checked if present.

    Returns
    -------
    list
        List of cell's names

    Examples
    --------
    >>> load_cells_names('/data/my_population.scool', 3)
    ['Cell1', 'Cell2', 'Cell3']
    >>> load_cells_names('/data/my_population.scool', ['Cell1', 'Cell2', 'Cell4])
    ['Cell1', 'Cell2']
    """
    
    try:
        if (cells_num is None and cells_names is None) or (cells_num is not None and cells_names is not None):
            raise ValueError("ValueError! You must pass either 'cells_num' or 'cells_names', but not both.")
    except ValueError as e:
        return str(e)

    with h5py.File(population_dir, 'r') as f:
        print('File successfully opened')
        print('Scool attributes:')
        for attr in f.attrs:
            print(attr, ': ', f.attrs[attr])

        cells_all = list(f.keys())

        if cells_num is not None:
            if len(cells_all) < cells_num:
                cells = cells_all 
            else:
                cells = cells_all[:cells_num]
        else:
            cells = [cell for cell in cells_all if cell in cells_names]

        return cells

def load_data(cell_dir: str, 
              chroms_list: Optional[list[str]] = None, 
              do_not_clean: Optional[bool] = False,
              normalization_percentile: Optional[int] = 90) -> tuple[np.ndarray, pd.Series]:
    """
    Extracts scHiC contact matrix and bins series describing this matrix. Data is loaded from cool file data source. If user works with composed population data within scool file the cell selection need to follow format: scool_file_directory/scool_file_name::cell_name. It picks only specified chromosomes. Functions remove_diag_plus and normalize_hic are included in it by default, but can be turned of with a parameter.
    
    Parameters
    ----------
    cell_dir : str
        Scool file directory with cell name.
    chroms_list : list of str
        List of chromosome names to loaded. Loads all if not specified.
    do_not_clean :  Optional[bool] = False
        True/False value controling whether normalization and main diagonal removal takes place.
    normalization_percentile : Optional[int] = 90
        Percentile of weakest contacts to be removed from contact matrices in.

    Returns
    -------
    numpy.ndarray
        n-dim numpy ndarray with scHiC contact matrix
    pandas.Series
        pandas series of bins describing contacts chromosome, start and end of chromatine fragment

    Examples
    --------
    >>> load_data('/data/my_population.scool::my_cell', ['chr1', 'chr2', 'chr3'])
    (array([[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.]]),
        chrom      start        end
    0    chr10          0    1000000
    1    chr10    1000000    2000000
    2    chr10    2000000    3000000
    3    chr10    3000000    4000000
    4    chr10    4000000    5000000
    ..     ...        ...        ...
    721  chr15   99000000  100000000
    722  chr15  100000000  101000000
    723  chr15  101000000  102000000
    724  chr15  102000000  103000000
    725  chr15  103000000  103494974
    """
    #load global cell selectors
    cell = cooler.Cooler(cell_dir)
    cell_contacts = cell.matrix(balance=False)
    cell_bins = cell.bins()

    if chroms_list == None:
        hic = cell_contacts[:]
        bin = cell_bins[:]
    else:
        chroms_num = len(chroms_list)
        hic_subsets_horizontal = [None] * chroms_num
        hic_subsets_vertical = [None] * chroms_num

        for row in range(chroms_num):
            chr_vertical = chroms_list[row]
            if row == 0:
                bin = cell_bins.fetch(chr_vertical)
            else:
                next_bin = cell_bins.fetch(chr_vertical)
                bin = pd.concat([bin, next_bin], ignore_index=True)
            for column in range(chroms_num):
                chr_horizontal = chroms_list[column]
                hic_subset = cell_contacts.fetch(chr_vertical, chr_horizontal)
                hic_subsets_horizontal[column] = hic_subset
            hic_subsets_vertical[row] = np.hstack(hic_subsets_horizontal)
        hic = np.vstack(hic_subsets_vertical)

    if do_not_clean == False:
        hic = remove_diag_plus(hic)
        hic = normalize_hic(hic, normalization_percentile)

    return hic, bin

def remove_diag_plus(matrix: np.ndarray) -> np.ndarray:
    """
    Sets main diagonal of matrix, and together with diagonals one above and below it to zero. This function is used as part of preprocessing to remove bins contacts on diagonal and those of neighboring bins which are not useful in chromatin reconstruction. It is used by default in load_data function.
    
    Parameters
    ----------
    matrix : np.ndarray
        Numpy square ndarray. 

    Returns
    -------
    np.ndarray
        Returns modified matrix.

    Examples
    --------
    >>> matrix = np.ndarray((4,4))
    >>> matrix[:] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> remove_diag_plus(matrix)
    >>> print(matrix)
    array([[0, 0, 3, 4],
           [0, 0, 0, 8],
           [9, 0, 0, 0],
           [13, 14, 0, 0]])
    """
    matrix_no_diag = matrix

    for i in range(matrix_no_diag.shape[0]):
        for j in range(matrix_no_diag.shape[1]):
            if i == j or i-1 == j or i+1 == j:
                matrix_no_diag[i][j] = 0

    return matrix_no_diag

def normalize_hic(hic: np.ndarray, p: int) -> np.ndarray:
    """
    Applies natural logarithm over matrix values and sets p-th percentile of lowest values to 0. This function is part of data preprocessing and is used by default in load_data function.
    
    Parameters
    ----------
    hic : np.ndarray
        Numpy square ndarray.
    p : 
        percentile controling which cells of matrix will be set to 0
        
    Returns
    -------
    np.ndarray
        Returns modified matrix.

    Examples
    --------
    >>> matrix = np.ndarray((4,4))
    >>> matrix[:] = [[0, 0, 3, 4], [0, 0, 0, 8], [9, 0, 0, 0], [13, 14, 0, 0]]
    >>> normalize_hic(matrix)
    >>> print(matrix)
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [13, 14, 0, 0]])
    """
    hic_normalized = np.log(hic + 1)
    flat = hic_normalized.flatten()
    thersh = np.percentile(flat, p)
    hic_normalized[hic_normalized < thersh] = 0
    
    return hic_normalized

def filter_poor_cells(population_dir: str, 
                      cells_names: list[str], 
                      chroms_list: list[str], 
                      min_contacts: Optional[int] = 4000, 
                      min_ratio: Optional[int] = 0.15, 
                      max_ratio: Optional[int] = 0.4, 
                      main_width: Optional[int] = 50,
                      normalization_percentile: Optional[int] = 90) -> np.array:
    """
    This function filters list of cells names and returns only cells which contact matrices have required minimal number of contacts as well as minimal and maximal ratio of long range contacts. Where long range contacts are defined as one's located beyond main_width distance of the central diagonal of matrix.  
    
    Parameters
    ----------
    population_dir: str
        Scool file directory containing cells data.
    cells_names: list[str]
        List of cells names to be filtered.
    chroms_list: list[str]
        List of chromosome names to be considered.
    min_contacts: Optional[int] = 4000
        Requires number of contacts to be higher than that.
    min_ratio: Optional[int] = 0.15
        Requires ratio of long range contacts to be higher than that.
    max_ratio: Optional[int] = 0.4
        Requires ratio of long range contacts to be lower than that.
    main_width: Optional[int] = 50
        Defines long range contacts. If contact is located on diagonal further than 50 diagonals up or down main diagonal of matrix it is constidered long range.
    normalization_percentile: Optional[int] = 90
        Optional parameter to control load_data function behaviour

    Returns
    -------
    numpy.array
        Array of cells names fulfilling filter requirements.

    Examples
    --------
    >>> filter_poor_cells('/data/my_population.scool', ['Cell1', 'Cell2', 'Cell3'], ['chr10', 'chr11', 'chr12'], min_contacts=4500, min_ratio=0.20, max_ratio=0.50, main_width=60)
    array(['Cell1'], dtype='<U38')
    """
    filtered_cells = [] 

    for cell_name in cells_names:
        hic, bins = load_data(population_dir+'::'+cell_name, chroms_list, normalization_percentile=normalization_percentile)
        contacts_num = np.count_nonzero(hic)
        if contacts_num < min_contacts:
            continue
        size = hic.shape[0]
        mask = np.abs(np.arange(size).reshape(-1, 1) - np.arange(size)) < main_width
        hic_masked = np.where(mask, 0, hic)
        contacts_num_no_main = np.count_nonzero(hic_masked)
        no_main_ratio = contacts_num_no_main / contacts_num
        if no_main_ratio < min_ratio or max_ratio < no_main_ratio:
            continue
        filtered_cells.append(cell_name)

    if len(filtered_cells) == 0:
        print("Warning: no cell fulfils quality requirements")
    return np.array(filtered_cells)

def sample_series(cells_names: list[str], 
                  labels: list[int], 
                  predictions: list[np.ndarray], 
                  series_size: int, 
                  debug: Optional[bool] = False) -> tuple[np.array, np.array, np.array]:
    """
    Samples desired number of cells from provided cells population. Keep input ratios of labels. Function returns three arrays with sampled cells names, labels and prediction values, all crucial for further steps.
    
    Parameters
    ----------
    cells_names: list[str]
        List of cells names to be filtered.
    labels: list[int]
        List of interphase labels given to the cells.
    predictions: list[np.ndarray]
        List of numpy ndarrays describing prediction.
    series_size: int
        Number of cells to be sampled.
    debug: bool
        Prints control info during execution. 

    Returns
    -------
    numpy.array
        Array of names of cells sampled in a series.
    numpy.array
        Array of labels of cells sampled in a series.
    numpy.array
        Array of prediction values of cells sampled in a series.

    Examples
    --------
    >>> print(filtered_population_names)
    ['Diploid_10_ACTGAGCG_AAGGCTAT_R1fastqgz'
    'Diploid_10_ACTGAGCG_CCTAGAGT_R1fastqgz'
    'Diploid_10_ACTGAGCG_CTATTAAG_R1fastqgz'
    'Diploid_10_ACTGAGCG_GAGCCTTA_R1fastqgz'
    'Diploid_10_ACTGAGCG_GCGTAAGA_R1fastqgz'
    'Diploid_10_ACTGAGCG_TCGACTAG_R1fastqgz'
    'Diploid_10_ATGCGCAG_AAGGCTAT_R1fastqgz'
    'Diploid_10_ATGCGCAG_CCTAGAGT_R1fastqgz'
    'Diploid_10_ATGCGCAG_CTATTAAG_R1fastqgz'
    'Diploid_10_ATGCGCAG_GCGTAAGA_R1fastqgz']
    >>> print(population_labels)
    [0 2 2 2 2 0 2 0 0 1]
    >>> print(population_predictions)
    [[0.50630042 0.2317334  0.26196618]
    [0.35175224 0.23726902 0.41097873]
    [0.35609426 0.02918836 0.61471738]
    [0.15329386 0.26029309 0.58641306]
    [0.31415687 0.17877836 0.50706477]
    [0.53080315 0.21029382 0.25890303]
    [0.32800212 0.03946776 0.63253011]
    [0.40546755 0.35522468 0.23930777]
    [0.37774182 0.26117363 0.36108455]
    [0.40079315 0.46998894 0.12921791]]
    >>> sample_series(filtered_population_names, population_labels, population_predictions, 4)
    (array(['Diploid_10_ACTGAGCG_TCGACTAG_R1fastqgz',
        'Diploid_10_ATGCGCAG_GCGTAAGA_R1fastqgz',
        'Diploid_10_ACTGAGCG_GCGTAAGA_R1fastqgz',
        'Diploid_10_ACTGAGCG_GAGCCTTA_R1fastqgz'], dtype='<U38'),
    array([0, 1, 2, 2]),
    array([[0.53080315, 0.21029382, 0.25890303],
            [0.40079315, 0.46998894, 0.12921791],
            [0.31415687, 0.17877836, 0.50706477],
            [0.15329386, 0.26029309, 0.58641306]]))
    """
    
    labels_types, labels_counts = np.unique(labels, return_counts=True)
    labels_ratio = labels_counts / len(cells_names)
    if debug:	
        print('labels_types', labels_types)
        print('labels_counts', labels_counts)
        print('labels_ratio', labels_ratio)

    new_labels_counts = (np.floor(labels_ratio * series_size)).astype(int)
    rand_num = series_size - np.sum(new_labels_counts)
    if debug:	
        print('new_labels_counts', new_labels_counts)
        print('rand_num', rand_num)

    labels_chance = np.cumsum(labels_ratio)
    labels_chance /= np.max(labels_chance)
    if debug:
        print('labels_chance', labels_chance)

    for i in range(rand_num):
        choice = np.searchsorted(labels_chance, np.random.random(1))
        new_labels_counts[choice] += 1
    
    if debug:	
        print('new_labels_counts', new_labels_counts)

    series_cells = []
    series_labels = []
    series_predictions = []

    for l in labels_types:
        idxs = np.where(labels == l)
        choice = random.sample(sorted(idxs[0]), new_labels_counts[l])

        series_cells.extend(cells_names[choice])
        series_labels.extend(labels[choice])
        series_predictions.extend(predictions[choice])

    return np.array(series_cells), np.array(series_labels), np.array(series_predictions)