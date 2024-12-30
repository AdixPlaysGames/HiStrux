import preparation as prep
from sklearn.neighbors import KDTree
import numpy as np
from scipy import ndimage as ndi

def get_enriched_series_data(series_names, series_predictions, population_names, population_dir, chroms_list, debug=False):
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

def get_series_data(series_names, population_dir, chroms_list, debug=False):
    hics = []
    bins = []
    
    series_num = len(series_names)
    for i in range(series_num):
        cell_name = series_names[i]
        hic, bin = prep.load_data(population_dir+'::'+cell_name, chroms_list)      
        hics.append(hic)
        bins.append(bin)

    return hics, bins

def get_supp_contacts(supp_cell, contacts_num, chroms, population_dir, debug=False):
    supp_hic, bins_neighbour = prep.load_data(population_dir+'::'+supp_cell, chroms)
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

def enrich_hic(ref_cell, supports, chroms, population_dir, debug=False):
    ref_hic, bins_ref = prep.load_data(population_dir+'::'+ref_cell, chroms)
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

def matrix_scalling(matrix, scale_ratio):
    order_of_interpolation = 3
    matrix_averaged = ndi.uniform_filter(input=matrix, size=scale_ratio)
    matrix_interpolated = ndi.zoom(input=matrix_averaged, zoom=1./scale_ratio, order=order_of_interpolation)

    return matrix_interpolated

def bins_scalling(bins, desired_num_bins, debug=False):
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

def generate_iterations_data(hics, bins, n):
    hics_scales = []
    bins_scales = []
    
    for c in range(len(hics)):
        hic = hics[c]
        bin = bins[c]

        hic_scales = [hic]
        bin_scales = [bin]

        for i in range(n-1):
            hic_scaled = matrix_scalling(hic, 5*(i+1))
            prep.normalize_hic(hic_scaled)
            prep.remove_diag_plus(hic_scaled)
            prep.normalize_hic(hic_scaled)
            hic_scales.append(hic_scaled)
            
            # print('required bins num: ', hic_scaled.shape[0])
            bin_scaled = bins_scalling(bin, hic_scaled.shape[0])
            bin_scales.append(bin_scaled) 

        hics_scales.append(hic_scales)
        bins_scales.append(bin_scales)

    return hics_scales, bins_scales
