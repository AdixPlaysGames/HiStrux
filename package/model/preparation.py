import selection as selec
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

def get_enriched_series_data(series_names: list[str], 
                             series_predictions: list[np.ndarray], 
                             population_names: list[str], 
                             population_dir: str, 
                             chroms_list: list[str], 
                             debug=False) -> tuple[list[np.ndarray], list[pd.Series]]:
    """
    Based on interphase prediction establishes cells neighbourhood and copies part of scHiC contacts from nearby cells to reduce contacts matrix sparsity.

    Parameters
    ----------
    series_names : list[str]
        List of names of cells in a series
    series_predictions : list[np.ndarray]
        List of arrays holding interphase prediction for each cell from series
    population_names : list[str]
        List of cells names from whole population
    population_dir : str
        Directory of scool file holding data for whole population.
    chroms_list : list[str]
        List of chroms names to be considered.

    Returns
    -------
    tuple of (list of numpy ndarray, pandas series)
        Touple holding two list with elements for each cell of the series. One with n-dim numpy arrays containing scHiC contact matrices and one with pandas series describing scHiC matrix bins.

    Examples
    --------
    """
    hics = []
    bins = []
    tree = KDTree(series_predictions)
    radius = 0.1

    neighbour_idx = tree.query_radius(series_predictions, r=radius)

    series_num = len(series_names)
    for i in range(series_num):
        ref_name = series_names[i]
        sup_names = population_names[neighbour_idx[i]]
        
        if debug:
            print('reference cell', i, ':', ref_name)
        
        new_hic, new_bins = enrich_hic(ref_name, sup_names, chroms_list, population_dir, debug)
        
        hics.append(new_hic)
        bins.append(new_bins)
        
    return hics, bins

def get_series_data(series_names: list[str], 
                    population_dir: str, 
                    chroms_list: list[str]) -> tuple[list[np.ndarray], list[pd.Series]]:
    """
    Extracts scHiC contacts matrix and it's bins description for each cell in series.

    Parameters
    ----------
    series_names : list[str]
        List of names of cells in a series
    population_dir : str
        Directory of scool file holding data for whole population.
    chroms_list : list[str]
        List of chroms names to be considered.

    Returns
    -------
    tuple of (list of numpy ndarray, pandas series)
        Touple holding two list with elements for each cell of the series. One with n-dim numpy arrays containing scHiC contact matrices and one with pandas series describing scHiC matrix bins.

    Examples
    --------
    """
    hics = []
    bins = []
    
    series_num = len(series_names)
    for i in range(series_num):
        cell_name = series_names[i]
        hic, bin = selec.load_data(population_dir+'::'+cell_name, chroms_list)      
        hics.append(hic)
        bins.append(bin)

    return hics, bins

def get_supp_contacts(supp_cell: str,
                      contacts_num: int,
                      chroms: list[str],
                      population_dir: str) -> np.ndarray:
    """
    Extracts desired number of contacts from cell randomly.

    Parameters
    ----------
    supp_cell : str
        Name of a choosen cell.
    contacts_num : int
        Number of contact to be extracted.
    chroms : list[str]
        List of chromosome names to be considered.
    population_dir : str
        Directory of scool file holding data for the cell.

    Returns
    -------
    numpy ndarray
        N-dim numpy array containing scHiC contact matrix.

    Examples
    --------
    """
    supp_hic, bins_neighbour = selec.load_data(population_dir+'::'+supp_cell, chroms)
    flat = supp_hic.flatten()
    shape = supp_hic.shape
    prob = flat / np.sum(flat)

    new_contacts = np.random.choice(len(prob), contacts_num, p = prob)
    new_contacts = np.unravel_index(new_contacts, shape)
    new_contacts = list(zip(new_contacts[0], new_contacts[1]))

    matrix = np.zeros(shape, dtype=float)
    for r, c in new_contacts:
        matrix[r, c] = supp_hic[r, c]
        matrix[c, r] = supp_hic[c, r]

    return matrix

def enrich_hic(ref_cell: str,
               supports: list[str],
               chroms: list[str],
               population_dir: str,
               debug=False) -> tuple[np.ndarray, pd.Series]:
    """
    Performs extraction of contacts from list of cells and adds those to the contacts matrix of choosen cell returning enriched cell contacts data in form of contacts matrix and series describing its bins. 

    Parameters
    ----------
    ref_cell : str
        Name of a choosen cell.
    supports : list[str]
        List of cells names from which contacts will be extracted
    chroms : list[str]
        List of chromosome names to be considered.
    population_dir : str
        Directory of scool file holding data for the cell.

    Returns
    -------
    tuple of (numpy.ndarray, pandas.Series)
        Touple of n-dim numpy array with scHiC contact matrix and pandas series of bins describing contacts chromosome, start and end of chromatine fragment.

    Examples
    --------
    """
    ref_hic, bins_ref = selec.load_data(population_dir+'::'+ref_cell, chroms)
    shape = ref_hic.shape
    new_hic = np.zeros(shape, dtype=float)
    new_contacts = np.zeros(shape, dtype=float)

    supports_num = len(supports)
    ref_contacts_num = np.count_nonzero(ref_hic)
    new_contacts_num = 0.1 * ref_contacts_num
    new_contacts_per_supp = np.ceil(new_contacts_num / supports_num).astype(int)
    
    if debug:
        print('reference contacts num', ref_contacts_num)

    for s in range(len(supports)):
        supp_cell = supports[s]
        new_contacts += get_supp_contacts(supp_cell, new_contacts_per_supp, chroms, population_dir, debug)
               
    new_hic = ref_hic + new_contacts

    if debug:
        print('support new contacts num', np.count_nonzero(new_contacts))
        print('new total contacts:', np.count_nonzero(new_hic))

    return new_hic, bins_ref

def matrix_scalling(matrix: np.ndarray,
                    scale_ratio: int) -> np.ndarray:
    """
    Scales down contacts matrix resolution by a given factor. Uses uniform filter and iterpolation of order 3.

    Parameters
    matrix : np.ndarray
        Numpy ndarray holding matrix of cells scHiC contacts.
    scale_ratio : int
        Scaling ratio defining factor of resolution lowering.
    
    Returns
    -------
    np.ndarray
        New scaled down contacts matrix.
    
    Examples
    --------
    """
    order_of_interpolation = 3
    matrix_averaged = ndi.uniform_filter(input=matrix, size=scale_ratio)
    matrix_interpolated = ndi.zoom(input=matrix_averaged, zoom=1./scale_ratio, order=order_of_interpolation)

    return matrix_interpolated

def bins_scalling(bins: pd.series,
                  desired_num_bins: int,
                  debug=False) -> pd.series:
    """
    Scales down bins description of contacts matrix to the choosen number of bins. Pics new bins begging and end as begging and end of bins merged and chromosome as mode of chromosomes of bins merged.

    Parameters
    ----------
    bins : pd.series
        Original series with bins description
    desired_num_bins : int
        Number of bins at the end of scalling.
    
    Returns
    -------
    pd.series
        New series containg merged bins in a desired number.
    
    Examples
    --------
    """
    # Calculate the scale ratio
    scale_ratio = len(bins) / desired_num_bins
    
    # Assign group indices carefully
    group_indices = (np.arange(len(bins)) / scale_ratio).astype(int)
    group_indices = np.minimum(group_indices, desired_num_bins - 1)  # Prevent out-of-bound indices

    # Group by the calculated indices
    grouped = bins.groupby(group_indices)

    # Aggregate using mode, min, and max
    bins_scaled = grouped.agg({
        'chrom': lambda x: x.mode().iloc[0],  # Mode of 'chrom'
        'start': 'min',                     # Minimum 'start'
        'end': 'max'                        # Maximum 'end'
    }).reset_index(drop=True)
    
    # Debugging output
    if debug:
        print('Original bins num:', len(bins))
        print('Desired bins num:', desired_num_bins)
        print('Scale ratio:', scale_ratio)
        print('Resulting bins num:', len(bins_scaled))
    
    return bins_scaled

def generate_iterations_data(hics: np.ndarray,
                             bins: pd.series,
                             n: int,) -> tuple[list[np.ndarray], list[pd.series]]:
    """
    Generates cell's scaled down contacts matrices and scaled down bins descriptions for them in a desired number.

    Parameters
    ----------
    hics : list[np.ndarray]
        List of n-dim numpy arrays containing contact matrices for cells in series.
    bins : list[pd.series]
        List of pandas serieses containing descriptions of bins for contact matrices for cells in series.
    n : int
        Number of iterations to be prepared.
    
    Returns
    -------
    tuple[list[np.ndarray], list[pd.series]]
        Touple with two lists holding scaled down cells contacts data. One with n-dim numpy arrays containing scaled down contact matrices and the second with scaled down bins descriptions. 
    
    Examples
    --------
    """
    hics_scales = []
    bins_scales = []
    
    for c in range(len(hics)):
        hic = hics[c]
        bin = bins[c]

        hic_scales = [hic]
        bin_scales = [bin]

        for i in range(n-1):
            hic_scaled = matrix_scalling(hic, 5*(i+1))
            selec.normalize_hic(hic_scaled)
            selec.remove_diag_plus(hic_scaled)
            selec.normalize_hic(hic_scaled)
            hic_scales.append(hic_scaled)
            
            # print('required bins num: ', hic_scaled.shape[0])
            bin_scaled = bins_scalling(bin, hic_scaled.shape[0])
            bin_scales.append(bin_scaled) 

        hics_scales.append(hic_scales)
        bins_scales.append(bin_scales)

    return hics_scales, bins_scales
