import numpy as np
import pandas as pd
import cooler
from typing import Optional
import random
import h5py

# data selection definitions

def load_cells_names(population_dir: str, 
                     num: int) -> list[str]:
    """
    Reads first n names of cells from scool file.

    Parameters
    ----------
    population_dir : str
        Scool file directory
    num : int
        Number of cell's names to be read

    Returns
    -------
    list
        List of cell's names

    Examples
    --------
    >>> load_cells_names('/data/my_population.scool', 3)
    ['Cell1', 'Cell2', 'Cell3']
    """
    with h5py.File(population_dir, 'r') as f:
        print('Plik otwarty pomyślnie')
        print('Atrybuty scool:')
        for attr in f.attrs:
            print(attr, ': ', f.attrs[attr])

        cells_all = list(f.keys())
        if len(cells_all) < num:
            cells = cells_all 
        else:
            cells = cells_all[:num]

        return cells

def load_data(cell_dir: str, 
              chroms_list: list[str], 
              normalization_percentile: Optional[int] = 90) -> tuple[np.ndarray, pd.Series]:
    """
    Loads a contacts matrix and a bins series of specified cell from .scool file

    This function extracts scHiC contact matrix and bins series describing this matrix. It picks only specified chromosomes. 
    
    Parameters
    ----------
    cell_dir : str
        Scool file directory with cell name
    chroms_list : list of str
        List of chromosome names to loaded 
    normalization_percentile : int
		Percentile of weakest contacts to be removed from contact matrices in.

    Returns
    -------
    tuple of (numpy.ndarray, pandas.Series)
        Touple of n-dim numpy array with scHiC contact matrix and pandas series of bins describing contacts chromosome, start and end of chromatine fragment.

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

    #set chroms range
    fetch_end = 0
    for chrom in chroms_list:
        fetch_end += cell_contacts.fetch(chrom).shape[0]
    
    #fetch desired part of HiC matrix
    hic = np.log(cell_contacts[0:fetch_end, 0:fetch_end]+1)
    remove_diag_plus(hic)
    normalize_hic(hic, normalization_percentile)
    bins_hic = cell.bins()[:]
    bins_hic = bins_hic[bins_hic['chrom'].isin(chroms_list)]

    return hic, bins_hic

def remove_diag_plus(matrix: np.ndarray) -> None:
    """
    Removes main diagonal of matrix, and both one above and below it.
    
    Parameters
    ----------
    matrix : np.ndarray
        Numpy square ndarray. 

    Returns
    -------
    None
        This function modifies provided array in place.

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
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j or i-1 == j or i+1 == j:
                matrix[i][j] = 0

def normalize_hic(hic: np.ndarray, p: int) -> None:
    """
    Sets p-th percentile of lowest values in hic matrix to 0.
    
    Parameters
    ----------
    hic : np.ndarray
        Numpy square ndarray. 

    Returns
    -------
    None
        This function modifies provided array in place.

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
    flat = hic.flatten()
    thersh = np.percentile(flat, p)
    hic[hic < thersh] = 0

def filter_poor_cells(population_dir: str, 
                      cells_names: list[str], 
                      chroms_list: list[str], 
                      min_contacts: Optional[int] = 4000, 
                      min_ratio: Optional[int] = 0.15, 
                      max_ratio: Optional[int] = 0.4, main_width: Optional[int] = 50) -> np.array:
    """
    Filters provided list of cells names with respect to contacts matricies quality. 

    This function filters list of cells names and returns only cells which contact matrices have required minimal number of contacts as well as minimal and maximal ratio of long range contacts. Where long range contacts are defined as one's located beyond main_width central diagonals of matrix.    
    
    Parameters
    ----------
    population_dir: str
		Scool file directory containing cells data
    cells_names: list[str]
		List of cells names to be filtered
    chroms_list: list[str]
        List of chromosome names to be considered
    min_contacts: Optional[int] = 4000
		Requires number of contacts to be higher than that.
    min_ratio: Optional[int] = 0.15
		Requires ratio of long range contacts to be higher than that.
    max_ratio: Optional[int] = 0.4
		Requires ratio of long range contacts to be lower than that.
    main_width: Optional[int] = 50
		Defines long range contacts. If contacts is located on diagonal further than 50 diagonals up or down main diagonal of matrix it is constidered long range.

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
        hic, bins = load_data(population_dir+'::'+cell_name, chroms_list)
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

    return np.array(filtered_cells)

def random_model(cells, phase_num):
    cells_num = len(cells)
    
    predictions = np.random.random((cells_num, phase_num))
    predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
    labels = np.argmax(predictions, axis=1)

    return labels, predictions

def sample_series(cells_names: list[str], 
                  labels: list[int], 
                  predictions: list[np.ndarray], 
                  series_size: int, 
                  debug=False) -> tuple[np.array, np.array, np.array]:
    """
    Samples desired number of cells from provided cells population. Keep input ratios of labels.
    
    Parameters
    ----------
    cells_names: list[str]
		List of cells names to be filtered
    labels: list[int]
		List of interphase labels given to the cells
    predictions: list[np.ndarray]
		List of numpy ndarrays describing prediction 
    series_size: int
		Number of cells to be sampled 

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
    if debug:	
        print('labels_types', labels_types)
        print('labels_counts', labels_counts)

    sample_ratio = series_size / len(cells_names)
    labels_ratio = labels_counts * sample_ratio
    if debug:	
        print('labels_ratio', labels_ratio)

    new_labels_counts = (np.floor(labels_ratio)).astype(int)
    rand_num = series_size - np.sum(new_labels_counts)

    labels_chance = np.cumsum(labels_ratio - new_labels_counts)
    labels_chance /= np.max(labels_chance)
    if debug:
        print('labels_chance', labels_chance)

    for i in range(rand_num):
        choice = np.searchsorted(labels_chance, np.random.random(1))
        new_labels_counts[choice] += 1

    # print('new_labels_counts', new_labels_counts)

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