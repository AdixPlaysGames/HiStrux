import HiStrux.reConstruct.selection as selec
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from typing import Optional, Union
import seaborn as sns
import matplotlib.pyplot as plt

def get_enriched_series_data(series_names: list[str], 
                             series_predictions: list[np.ndarray], 
                             population_names: list[str], 
                             population_dir: str,
                             chroms_list: Optional[list[str]] = None, 
                             debug: Optional[bool] = False) -> tuple[list[np.ndarray], list[pd.Series]]:
    """
    This function enriches contact matrices of cells in the series by perforeming search in KDTree to find most similar cells in population based on the interphase prediction.
    
    Parameters
    ----------
    series_names : list[str]
        List of cells names to be enriched.
    series_predictions : list[np.ndarray]
        List of prediction np.array vectors.
    population_names : list[str]
        List of cells names from whole population.
    population_dir : str
        Scool file directory.
    chroms_list : Optional[list[str]] = None
        List of chromosome names to loaded. Loads all if not specified.
    debug: Optional[bool] = False
        Prints control info during execution. 

    Returns
    -------
    list[np.ndarray]
        list of enriched n-dim numpy ndarrays with scHiC contact matricies for each cell from the series.
    list[pd.Series]]
        list of bins description for contact maps for each cell from the series.

    Examples
    --------
    >>> get_enriched_series_data(['Diploid_10_ACTGAGCG_TCGACTAG_R1fastqgz'], 
                                 [[0.46271356 0.35939344 0.05374532 0.12414767]], 
                                 [['Diploid_10_ACTCGCTA_TCGACTAG_R1fastqgz'
                                    'Diploid_10_ACTGAGCG_AAGGCTAT_R1fastqgz'
                                    'Diploid_10_ACTGAGCG_CCTAGAGT_R1fastqgz'
                                    'Diploid_10_ACTGAGCG_CTATTAAG_R1fastqgz'
                                    'Diploid_10_ACTGAGCG_GAGCCTTA_R1fastqgz'
                                    'Diploid_10_ACTGAGCG_GCGTAAGA_R1fastqgz'
                                    'Diploid_10_ACTGAGCG_TCGACTAG_R1fastqgz'
                                    'Diploid_10_ATGCGCAG_AAGGCTAT_R1fastqgz'
                                    'Diploid_10_ATGCGCAG_CCTAGAGT_R1fastqgz'
                                    'Diploid_10_ATGCGCAG_CTATTAAG_R1fastqgz'
                                    'Diploid_10_ATGCGCAG_GAGCCTTA_R1fastqgz'
                                    'Diploid_10_ATGCGCAG_GCGTAAGA_R1fastqgz'], 
                                  '../../../data/nagano2017/nagano_1MB_raw.scool', 
                                  ['chr1', 'chr2', 'chr3', 'chr4', 'chr5'])
    ([array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          0.        ],
         [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          0.        ],
         [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          0.        ],
         ...,
         [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          1.38629436],
         [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          0.        ],
         [0.        , 0.        , 0.        , ..., 1.38629436, 0.        ,
          0.        ]])],
    [    chrom      start        end
    0    chr1          0    1000000
    1    chr1    1000000    2000000
    2    chr1    2000000    3000000
    3    chr1    3000000    4000000
    4    chr1    4000000    5000000
    ..    ...        ...        ...
    844  chr5  148000000  149000000
    845  chr5  149000000  150000000
    846  chr5  150000000  151000000
    847  chr5  151000000  152000000
    848  chr5  152000000  152537259
    """
    hics = []
    bins = []
    tree = KDTree(series_predictions)
    radius = 0.1

    neighbour_idx = np.array(tree.query_radius(series_predictions, r=radius))

    series_num = len(series_names)
    for i in range(series_num):
        ref_name = series_names[i]
        sup_names = [population_names[idx] for idx in neighbour_idx[i]]
        
        if debug:
            print('reference cell', i, ':', ref_name)
            print('support names', sup_names)
        
        new_hic, new_bins = enrich_hic(ref_name, sup_names, population_dir, chroms_list, debug)
        
        hics.append(new_hic)
        bins.append(new_bins)
        
    return hics, bins

def get_series_data(series_names: list[str],
                    population_dir: str,
                    chroms_list: Optional[list[str]] = None) -> tuple[list[np.ndarray], list[pd.Series]]:
    """
    Extracts data about cells from the series using load_data function.
    
    Parameters
    ----------
    series_names: list[str]
        List of cells names.
    population_dir: str
        Scool file directory containing cells data.
    chroms_list: Optional[list[str]] = None
        List of chromosome names to loaded. Loads all if not specified.
    
    Returns
    -------
    list[np.ndarray]
        list of n-dim numpy ndarrays with scHiC contact matricies for each cell from the series
    list[pd.Series]]
        list of bins description for contact maps for each cell from the series
    
    Examples
    --------
    >>> get_series_data(['Diploid_10_ACTGAGCG_TCGACTAG_R1fastqgz'],
                        '../../../data/nagano2017/nagano_1MB_raw.scool',
                        ['chr1', 'chr2', 'chr3', 'chr4', 'chr5'])
    ([array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          0.        ],
         [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          0.        ],
         [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          0.        ],
         ...,
         [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          1.38629436],
         [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
          0.        ],
         [0.        , 0.        , 0.        , ..., 1.38629436, 0.        ,
          0.        ]])],
    [    chrom      start        end
    0    chr1          0    1000000
    1    chr1    1000000    2000000
    2    chr1    2000000    3000000
    3    chr1    3000000    4000000
    4    chr1    4000000    5000000
    ..    ...        ...        ...
    844  chr5  148000000  149000000
    845  chr5  149000000  150000000
    846  chr5  150000000  151000000
    847  chr5  151000000  152000000
    848  chr5  152000000  152537259
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
                      population_dir: str,
                      chroms_list: Optional[list[str]] = None) -> np.ndarray:
    """
    Extracts required number of random contacts from specified cell's HiC matrix. Since contacts need to be simmetrical most of them will be doubled to the other side of HiC matrix. If you want to determine total number of elements extracted from cell's contact matrix set contacts_num parameter to half of that number.
    
    Parameters
    ----------
    supp_cell: str
        Cell name.
    contacts_num: int
        Number of contacts to be extracted from the cell's HiC matrix.
    population_dir: str
        Scool file directory containing cells data.
    chroms_list: Optional[list[str]] = None
        List of chromosome names to loaded. Loads all if not specified.
        
    Returns
    -------
    np.ndarray
        HiC matrix with extracted contacts
        
    Examples
    --------
    >>> get_supp_contacts('Diploid_10_ACTCGCTA_TCGACTAG_R1fastqgz',
                           100,
                           './../../data/nagano2017/nagano_1MB_raw.scool',
                           ['chr1', 'chr2', 'chr3', 'chr4', 'chr5'])
    array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
    >>> np.count_nonzero(get_supp_contacts('Diploid_10_ACTCGCTA_TCGACTAG_R1fastqgz',
                           100,
                           './../../data/nagano2017/nagano_1MB_raw.scool',
                           ['chr1', 'chr2', 'chr3', 'chr4', 'chr5']))
    198
    """
    supp_hic, bins_neighbour = selec.load_data(population_dir+'::'+supp_cell, chroms_list)
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
               population_dir: str,
               chroms_list: Optional[list[str]] = None, 
               extraction_fraction: Optional[int] = 0.1,
               debug: Optional[bool] = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Enriches referenced cell with contacts from support cells. Contacts are added equaly from all support cells and their number is set with a parameter as fraction of referenced cell contact numbers. 
    
    Parameters
    ----------
    ref_cell: str
        Referencial cell name.
    supports: list[str]
        List of support cells names.
    population_dir: str
        Scool file directory containing cells data.
    chroms_list: Optional[list[str]] = None
        List of chromosome names to loaded. Loads all if not specified.
    extraction_fraction: Optional[int] = 0.1
        Fraction of oryginal contacts that is supposed to be added
    debug: Optional[bool] = False
        Prints control info during execution. 
        
    Returns
    -------
    list[np.ndarray]
        Enriched n-dim numpy ndarray with scHiC contact matrix of referenced cell.
    list[pd.Series]
        Bins description for contact map of referenced cell.
    
    Examples
    --------
    >>> enrich_hic('Diploid_10_ACTGAGCG_TCGACTAG_R1fastqgz', 
                   ['Diploid_10_ACTCGCTA_TCGACTAG_R1fastqgz','Diploid_10_ACTGAGCG_AAGGCTAT_R1fastqgz'], 
                   '../../../data/nagano2017/nagano_1MB_raw.scool', 
                   ['chr1', 'chr2', 'chr3', 'chr4', 'chr5'], 
                   debug=True)
    reference contacts num 9782
    support new contacts num 1816
    new total contacts: 11292
    (array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        ...,
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         1.38629436],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 1.38629436, 0.        ,
         0.        ]]),
        chrom      start        end
    0    chr1          0    1000000
    1    chr1    1000000    2000000
    2    chr1    2000000    3000000
    3    chr1    3000000    4000000
    4    chr1    4000000    5000000
    ..    ...        ...        ...
    844  chr5  148000000  149000000
    845  chr5  149000000  150000000
    846  chr5  150000000  151000000
    847  chr5  151000000  152000000
    848  chr5  152000000  152537259
    """
    ref_hic, bins_ref = selec.load_data(population_dir+'::'+ref_cell, chroms_list)
    shape = ref_hic.shape
    new_hic = np.zeros(shape, dtype=float)
    new_contacts = np.zeros(shape, dtype=float)

    supports_num = len(supports)
    ref_contacts_num = np.count_nonzero(ref_hic)
    new_contacts_num = extraction_fraction * ref_contacts_num
    new_contacts_per_supp = np.ceil(new_contacts_num / supports_num).astype(int)
    
    if debug:
        print('reference contacts num', ref_contacts_num)

    for s in range(len(supports)):
        supp_cell = supports[s]
        new_contacts += get_supp_contacts(supp_cell, new_contacts_per_supp, population_dir, chroms_list)
               
    new_hic = ref_hic + new_contacts

    if debug:
        print('support new contacts num', np.count_nonzero(new_contacts))
        print('new total contacts:', np.count_nonzero(new_hic))

    return new_hic, bins_ref

def matrix_scalling(matrix: np.ndarray, 
                    scale: int) -> np.ndarray:
    """
    Scales down HiC matrix provided by a scale ratio provided by taking a mean of values over a window of size equal to scale.
    
    Parameters
    ----------
    matrix: np.ndarray
        Oryginal contact matrix to be scaled.
    scale: int
        Integer number determining ratio of scaling.
        
    Returns
    -------
    np.ndarray
        Scaled down contact matrix.
        
    Examples
    --------
    >>> print(hic.shape)
    (849, 849)
    >>> hic_scaled = matrix_scalling(hic, 8)
    >>> print(hic_scaled.shape)
    (106, 106)
    """
    rows, cols = matrix.shape
    new_rows = (rows ) // scale  
    new_cols = (cols ) // scale  
    
    downscaled_matrix = np.zeros((new_rows, new_cols))
    
    for i in range(new_rows):
        for j in range(new_cols):
            row_start, row_end = i * scale, min((i + 1) * scale, rows)
            col_start, col_end = j * scale, min((j + 1) * scale, cols)

            block = matrix[row_start:row_end, col_start:col_end]

            downscaled_matrix[i, j] = block.mean()

    return downscaled_matrix

def bins_scalling(bins: pd.Series, 
                  desired_num_bins: int, 
                  debug: Optional[bool] = False) -> pd.Series:
    """
    Scales down number of bins in the bins description pandas Series provided. Groups bins to achieve desired end number by taking mode of chromosomes assigned of the original bins as well as minimal start and maximal end.
    
    Parameters
    ----------
    bins: pd.Series
        Original bins description.
    desired_num_bins: int
        Number of bins to be returned at the end.
    debug: Optional[bool] = False
        Prints control info during execution. 

    Returns
    -------
    pd.Series
        Scaled down bins description pandas Series.
        
    Examples
    --------
    >>> print(len(bin))
    849
    >>> bin_scaled = bins_scalling(bin, 106)
    >>> print(len(bin_scaled))
    106
    """
    scale_ratio = len(bins) / desired_num_bins
    
    group_indices = (np.arange(len(bins)) / scale_ratio).astype(int)
    group_indices = np.minimum(group_indices, desired_num_bins - 1) 

    grouped = bins.groupby(group_indices)

    bins_scaled = grouped.agg({
        'chrom': lambda x: x.mode().iloc[0],  
        'start': 'min',                     
        'end': 'max'                   
    }).reset_index(drop=True)
    
    # Debugging output
    if debug:
        print('Original bins num:', len(bins))
        print('Desired bins num:', desired_num_bins)
        print('Scale ratio:', scale_ratio)
        print('Resulting bins num:', len(bins_scaled))
    
    return bins_scaled

def generate_iterations_data(hics: list[np.ndarray], 
                             bins: list[pd.Series], 
                             n: int,
                             scale: int, 
                             p: Optional[int] = 90) -> tuple[list[np.ndarray, list[pd.Series]]]:
    """
    Scales down number of bins and size of contact matricies of all HiC maps and bins descriptions provided to achieve desired number of verisons. Parameter n controls number of data sets to be reached at the end, where first data set is leaved as original size data. After each scaling remove_diag_plus and normalize_hic functions are applied again.
    
    Parameters
    ----------
    hics : list[pd.Series]
        Original HiC matricies of cells.
    bins : list[pd.Series]
        Original bins descriptions of cells.
    n : int
        Number of verisions to be returned at the end.
    p : Optional[int] = 90
        Controls normalize_hic function behaviour.

    Returns
    -------
    pd.Series
        Scaled down bins description pandas Series.
        
    Examples
    --------
    >>> generate_iterations_data(hics, bins, 5)
    ([[array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          ...,
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           1.60943791],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 1.60943791, 0.        ,
           0.        ]]),
   array([[0.        , 0.        , 0.2138843 , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.2138843 , 0.        , 0.        , ..., 0.02772589, 0.        ,
           0.        ],
          ...,
          [0.        , 0.        , 0.02772589, ..., 0.        , 0.        ,
           0.15955936],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.15955936, 0.        ,
           0.        ]])],
  [array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          ...,
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.69314718],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.69314718, 0.        ,
           0.        ]]),
   array([[0.        , 0.        , 0.14755518, ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.14755518, 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          ...,
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.78675043],
          [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
           0.        ],
          [0.        , 0.        , 0.        , ..., 0.78675043, 0.        ,
           0.        ]])]],
    [[    chrom      start        end
    0    chr1          0    1000000
    1    chr1    1000000    2000000
    2    chr1    2000000    3000000
    3    chr1    3000000    4000000
    4    chr1    4000000    5000000
    ..    ...        ...        ...
    844  chr5  148000000  149000000
    845  chr5  149000000  150000000
    846  chr5  150000000  151000000
    847  chr5  151000000  152000000
    848  chr5  152000000  152537259
    
    [849 rows x 3 columns],
        chrom      start        end
    0    chr1          0    6000000
    1    chr1    6000000   11000000
    2    chr1   11000000   16000000
    3    chr1   16000000   21000000
    4    chr1   21000000   26000000
    ..    ...        ...        ...
    164  chr5  128000000  133000000
    165  chr5  133000000  138000000
    166  chr5  138000000  143000000
    167  chr5  143000000  148000000
    168  chr5  148000000  152537259
    
    [169 rows x 3 columns]],
    [    chrom      start        end
    0    chr1          0    1000000
    1    chr1    1000000    2000000
    2    chr1    2000000    3000000
    3    chr1    3000000    4000000
    4    chr1    4000000    5000000
    ..    ...        ...        ...
    844  chr5  148000000  149000000
    845  chr5  149000000  150000000
    846  chr5  150000000  151000000
    847  chr5  151000000  152000000
    848  chr5  152000000  152537259
    
    [849 rows x 3 columns],
        chrom      start        end
    0    chr1          0    6000000
    1    chr1    6000000   11000000
    2    chr1   11000000   16000000
    3    chr1   16000000   21000000
    4    chr1   21000000   26000000
    ..    ...        ...        ...
    164  chr5  128000000  133000000
    165  chr5  133000000  138000000
    166  chr5  138000000  143000000
    167  chr5  143000000  148000000
    168  chr5  148000000  152537259
    
    [169 rows x 3 columns]]])
    """
     
    hics_scales = []
    bins_scales = []
    
    for c in range(len(hics)):
        hic = hics[c]
        bin = bins[c]

        hic_scales = [hic]
        bin_scales = [bin]

        for i in range(n-1):
            hic_scaled = matrix_scalling(hic, scale**(i+1))
            selec.remove_diag_plus(hic_scaled)
            selec.normalize_hic(hic_scaled, p)
            hic_scales.append(hic_scaled)
            
            # print('required bins num: ', hic_scaled.shape[0])
            bin_scaled = bins_scalling(bin, hic_scaled.shape[0])
            bin_scales.append(bin_scaled) 

        hics_scales.append(hic_scales)
        bins_scales.append(bin_scales)

    return hics_scales, bins_scales

def check_iterations_setup(hics_scales: list[np.ndarray], 
                           cell_idx: int, 
                           orientation: Optional[str] = 'Horizontal') -> None:
    """
    For the cell's index in the hic_scales list, prints number of bins and number of non zero values in the hic matrix and plots each version of hic matrix of this cell. Allowed values of orientation parameter are 'Horizontal' and 'Vertical'.
    
    Parameters
    ----------
    hics_scales : list[str]
        List of verisions of hic matrix. 
    cell_idx : int
        Index of cell that is to be checked from hic_scales list.
    orientation :  Optional[str] = 'Horizontal'
        Orientation of output plot.

    Returns
    -------
    list[np.ndarray]
        list of enriched n-dim numpy ndarrays with scHiC contact matricies for each cell from the series.
    list[pd.Series]
        list of bins description for contact maps for each cell from the series.

    Examples
    --------
    >>> check_iterations_setup(hics_scales, 1)
    computational load:
    iteration 1
    number of particles: 169
    number of bonds: 5154
    iteration 2
    number of particles: 849
    number of bonds: 22498
    """
    print('computational load:')
    iteration_num = len(hics_scales[cell_idx])
    for i in range(iteration_num-1, -1, -1):
        print('iteration', iteration_num-i)
        particles_num = hics_scales[cell_idx][i].shape[0]
        num_contacts = np.count_nonzero(hics_scales[cell_idx][i])
        print('number of particles:', particles_num)
        print('number of bonds:', num_contacts)
    
    if orientation == 'Horizontal':
        fig, axes = plt.subplots(1, iteration_num, figsize=(30, 20/iteration_num))
        for i in range(1, iteration_num+1):
            sns.heatmap(hics_scales[0][i-1], ax=axes[i-1])
            axes[i-1].set_title(f'iteration num: {i}')
    elif orientation == 'Vertical':
        fig, axes = plt.subplots(iteration_num, 1, figsize=(30/iteration_num, 25))
        for i in range(1, iteration_num+1):
            sns.heatmap(hics_scales[0][i-1], ax=axes[i-1])
            axes[i-1].set_title(f'iteration num: {i}')
    else:
        return 'No proper orientation specified.'
