import numpy as np
import pandas as pd
import gsd.hoomd
import hoomd
import itertools
from typing import Optional, Union
import os
import HiStrux.reConstruct.review as rev

def generate_new_particles(trajectory_dir: str, 
                           next_bin: pd.Series, 
                           debug: Optional[bool] = False) -> list[np.array]:
    """
    Calculates new set of particles positions to achieve description of bins provided by next_bin while staying within chains position taken from last frame of simulation record provided by trajectory_dir. New particles are spread evenly along chains length.
    
    Parameters
    ----------
    trajectory_dir : str
        String specifying location of .gsd file holding in it end state of a simulation which resolution user wants to expand.  
    next_bin : pd.Series
        Bins description which will determine number of particles returned
    debug : Optional[bool] = False
    
    Returns
    -------
    list[np.array]
        List of 3 dimensional np.arrays holding new positions of particles.

    Examples
    --------
    >>> generate_new_particles('./test_frame.gsd', bins_scales[0][0], debug=True)
    chromosom_legend {0: 'chr1', 1: 'chr2', 2: 'chr3', 3: 'chr4', 4: 'chr5'}
    chrom generated: chr1
    chrom_start_num: 39
    chrom_end_num: 198
    chrom_add_num: 159
    old_idx [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    24 25 26 27 28 29 30 31 32 33 34 35 36 37]
    old_idx num 38
    new_idxs [ 0  0  0  0  0  1  1  1  1  2  2  2  2  2  3  3  3  3  3  3  4  4  4  4
    5  5  5  6  6  6  6  6  6  7  7  7  7  7  8  8  8  8  8  8  8  9  9  9
    9  9  9 10 10 10 10 10 10 11 11 11 11 11 11 12 12 12 12 12 12 13 13 13
    13 13 13 13 14 14 14 14 14 14 14 15 15 15 16 16 16 17 17 17 17 17 17 17
    18 18 18 18 18 18 18 19 19 19 19 19 19 20 20 20 20 20 20 21 21 21 21 21
    22 22 22 22 22 22 22 23 23 23 24 24 24 24 24 24 25 25 26 26 26 26 26 26
    26 27 27 27 27 27 27 27 27 28 28 29 29 29 29 29 29 30 30 30 30 30 30 31
    31 31 31 31 32 33 33 33 33 33 33 34 34 34 35 35 35 35 35 35 36 36 36 37
    37 37 37 37 38]
    new_idxs num 197
    starting from [[ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
    [ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
    [ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
    [ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
    [ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
    [-1.7052641e+00  5.7458645e-01 -3.0677056e+00]
    [-1.7052641e+00  5.7458645e-01 -3.0677056e+00]
    ...
    [60.49659337 14.14804498 45.72595499]
    [61.00875445 14.45202941 46.11389353]
    [61.47748046 14.72086156 46.30976203]
    [61.75637381 14.83606192 45.66621001]]
    Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
    array([[ 0.        ,  0.        ,  0.        ],
        [-0.31627893,  0.10656977, -0.56897386],
        [-0.63255787,  0.21313953, -1.13794771],
        ...,
        [61.00875445, 14.45202941, 46.11389353],
        [61.47748046, 14.72086156, 46.30976203],
        [61.75637381, 14.83606192, 45.66621001]])
    """
    
    with gsd.hoomd.open(trajectory_dir) as f:
        frame = f[-1]
    
    particles_position = frame.particles.position
    particles_types = frame.particles.types
    # print(particles_types)
    particles_types_id = frame.particles.typeid
    chromosom_legend = dict(zip(np.unique(particles_types_id), particles_types))

    next_value_counts = next_bin['chrom'].value_counts()
    # next_value_counts = next_value_counts[next_value_counts > 0].sort_index()
    # print(next_value_counts)

    new_particles_position = []
    for c in range(len(particles_types)):
        chrom = particles_types[c]
        chrom_end_num = next_value_counts[chrom]

        concat_particles_info = zip(particles_position, particles_types_id)
        filtered_particles_info = [(position, type_id) for position, type_id in concat_particles_info if chromosom_legend[type_id] == chrom]
        filtered_particles_position, filtered_particles_types_id = zip(*filtered_particles_info)
        filtered_particles_position = np.array(list(filtered_particles_position))
        filtered_particles_types_id = np.array(list(filtered_particles_types_id))

        chrom_start_num = len(filtered_particles_position)
        chrom_add_num = chrom_end_num - chrom_start_num

        if debug:
            print('chromosom_legend', chromosom_legend)
            print('chrom generated:', chrom)
            print('chrom_start_num:', chrom_start_num)
            print('chrom_end_num:', chrom_end_num)
            print('chrom_add_num:', chrom_add_num)

        bonds_position_start = filtered_particles_position[:-1]
        bonds_position_end = filtered_particles_position[1:]
        bonds_length = np.linalg.norm(bonds_position_end - bonds_position_start, axis=1)
        bonds_cumsum = np.cumsum(bonds_length)
        new_dist = sum(bonds_length) / (chrom_end_num - 1)
        new_bonds_cumsum = np.cumsum([new_dist]*(chrom_end_num - 1))

        old_idx = np.searchsorted(bonds_cumsum, bonds_cumsum)
        new_idxs = np.searchsorted(bonds_cumsum, new_bonds_cumsum)

        bonds_dir = (bonds_position_end - bonds_position_start) / np.array([bonds_length]).T
        bonds_dir = np.vstack([bonds_dir, [0,0,0]])
        segment_dist = np.hstack([0, bonds_cumsum])
        dist_range = np.array(range(1, chrom_end_num))

        new_chrom_particles_position = filtered_particles_position[new_idxs] + bonds_dir[new_idxs] * np.array([new_dist * dist_range - segment_dist[new_idxs]] * 3).T
        new_chrom_particles_position = np.vstack((filtered_particles_position[0], new_chrom_particles_position))

        if debug:
            print('old_idx', old_idx)
            print('old_idx num', len(old_idx))
            print('new_idxs', (new_idxs))
            print('new_idxs num', len(new_idxs))
            print('starting from', filtered_particles_position[new_idxs])
            print('going towards', bonds_dir[new_idxs])
            print('going for', np.array([new_dist * dist_range - segment_dist[new_idxs]] * 3).T)
            print('new_chrom_particles_position', new_chrom_particles_position)

        new_particles_position.append(new_chrom_particles_position)

    new_particles_position = np.vstack(new_particles_position)
    return new_particles_position

def generate_initial_positions(particles_count: int,
                               radius: int,
                               box_size: int) -> list[np.array]:
    """
    Generates initial positions of particles to be used in simulation initiation by using random walk.
    
    Parameters
    ----------
    particles_count : int
        Number of particles to be generated
    radius : int
        Size of the step when performing random walk. Will determine distance between generated positions.
    box_size : int
        Constraint on positions generated ensuring no particle will be given position outside of the simulation box.
    
    Returns
    -------
    list[np.array]
        List of 3 dimensional np.arrays holding initial positions of particles.

    Examples
    --------
    >>> generate_initial_positions(169, 1, 50)
    array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-5.72993301e-01, -9.16848623e-01, -5.40761736e-01],
       [ 7.62070111e-02, -2.05794635e-01, -6.13924149e-01],
       [-6.81294849e-01,  5.99844209e-01,  3.60949628e-02],
       [ 1.63254061e-01,  1.54804675e+00, -6.80280606e-02],
       [-7.19805624e-01,  1.82536159e+00, -8.99686964e-01],
       [-1.35698088e+00,  1.44191747e+00, -1.06326082e+00],
       [-5.27316493e-01,  1.71036633e+00, -1.75043193e+00],
       [ 2.65728128e-01,  2.13923773e+00, -2.45423359e+00],
       [-1.31685503e-01,  2.67329759e+00, -2.90523244e+00],
       [ 8.60127891e-02,  2.75981073e+00, -2.94916762e+00],
       [-1.03529561e-01,  3.14450165e+00, -3.64494361e+00],
       [ 2.88306448e-01,  3.49293476e+00, -4.02396203e+00],
       [-6.93212701e-01,  3.63064657e+00, -4.61636354e+00],
       [ 1.93876078e-01,  2.95197476e+00, -4.11671417e+00],
       [ 6.59374221e-01,  3.17923378e+00, -4.39934160e+00],
       [ 1.12749628e+00,  3.92942818e+00, -3.52535340e+00],
       [ 2.34325781e-01,  4.59357635e+00, -2.75406365e+00],
       [ 9.18592357e-01,  5.40573265e+00, -3.04404011e+00],
       [ 1.10497319e+00,  4.63750138e+00, -3.99706130e+00],
       [ 1.70790741e+00,  4.44501844e+00, -4.66123650e+00],
       [ 2.55263001e+00,  4.21102206e+00, -5.51010015e+00],
       [ 2.27172913e+00,  4.08972764e+00, -6.00980706e+00],
       [ 3.23212767e+00,  4.85937866e+00, -5.43301697e+00],
       [ 3.96052339e+00,  5.84950415e+00, -4.45938346e+00],
    ...
       [ 8.93377412e+00, -1.41604450e+01, -3.29538790e+00],
       [ 9.61538778e+00, -1.50483063e+01, -2.52155734e+00],
       [ 9.12140479e+00, -1.56316652e+01, -1.81626101e+00],
       [ 8.66009048e+00, -1.65017794e+01, -1.00518598e+00],
       [ 9.31116976e+00, -1.65338110e+01, -1.38459180e-01]])
    """
    positions = np.array([[0,0,0]])

    i = 1
    while i < particles_count:
        random = np.random.rand(1, 3)
        direction = (random * 2 * radius) - radius
        new_position = positions[i-1] + direction

        # positions = np.concatenate((positions, new_position), axis=0)
        if (np.abs(new_position) < box_size).all():
            positions = np.concatenate((positions, new_position), axis=0)
            i+=1

    return positions

def frame_initiation(matrix: np.ndarray, 
                     bins: pd.Series, 
                     frame_name: str, 
                     new_particles_position: Optional[list[np.array]] = None, 
                     debug: Optional[bool] = False) -> list[Union[hoomd.md.bond.Harmonic, hoomd.md.pair.Gaussian]]:
    """
    Initializes first frame of the simulation. Defines particles types and positions as well as forces in the simulation(chain_force, contact_force, colision_force, colision_force_weak). When debug parameter is set to False saves frame to the .gsd file and return list of hoomd.md objects specifying forces taking place in the simulation.
    
    Parameters
    ----------
    matrix : np.ndarray
        HiC matrix on which simulation will be based.
    bins : pd.Series
        Bins description on which simulation will be based.
    frame_name : str
        Filename of .gsd in which frame will be saved to.
    new_particles_position : Optional[list[np.array]] = None
        Positions in which particles will be placed. When no positions specified generate_initial_positions function will be used.
    debug: Optional[bool] = False
        Prints control info during execution. 

    Returns
    -------
    list[np.ndarray]
        list of enriched n-dim numpy ndarrays with scHiC contact matricies for each cell from the series.
    list[pd.Series]
        list of bins description for contact maps for each cell from the series.

    Examples
    --------
    >>> frame_initiation(hics_scales[0][1], bins_scales[0][1], './test_frame.gsd', debug=True)
    contact_count: 4854
    len(bins.chrom): 5
    particles:
    num  169
    postitions num  169
    types  ['chr1', 'chr2', 'chr3', 'chr4', 'chr5']
    types num  169
    bonds:
    num  5018
    types  ['chr1-chr1', 'chr2-chr2', 'chr3-chr3', 'chr4-chr4', 'chr5-chr5', 'contact']
    typeid [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    chromosom_legend {'chr1': 0, 'chr2': 1, 'chr3': 2, 'chr4': 3, 'chr5': 4}
    unique types num [0, 1, 2, 3, 4, 5]
    types num  5018
    groups num  5018
    
    [<hoomd.md.bond.Harmonic at 0x7f2cddbc3f40>,
    <hoomd.md.bond.Harmonic at 0x7f2cddbd4f70>,
    <hoomd.md.pair.pair.Gaussian at 0x7f2cddbc2f10>,
    <hoomd.md.pair.pair.Gaussian at 0x7f2cddbc2f10>]
    """
    
    chrom_name_to_id = {element : i for i, element in enumerate(pd.unique(bins['chrom']))}
    box_size = 150

    # particles parameters extraction
    particles_count = matrix.shape[0]
    particles_chroms = bins.chrom
    particles_id = particles_chroms.map(chrom_name_to_id).astype(int).reset_index(drop=True)
    if new_particles_position is None:
        particles_positions = generate_initial_positions(particles_count, box_size/100, box_size)
    else:
        particles_positions = new_particles_position

    # forces parameters setting
    harmonic = hoomd.md.bond.Harmonic()
    harmonic_params = dict(k = 2000.0, r0 = 1.0)

    harmonic2 = hoomd.md.bond.Harmonic()
    harmonic2_params = dict(k = 2000.0, r0 = 1.5)

    gaussian_params = dict(epsilon = 100.0, sigma = 1)
    
    gaussian_nlist_params = dict(buffer = 1, default_r_cut=3.5)
    contact_nlist = hoomd.md.nlist.Cell(gaussian_nlist_params['buffer'])
    gaussian = hoomd.md.pair.Gaussian(contact_nlist, gaussian_nlist_params['default_r_cut'])

    # chain bonds definition
    bond_types = []
    for ch in particles_chroms.unique():
        bond_target = f'{ch}' + '-' + f'{ch}' 
        bond_types.append(bond_target)

    # frame declaration
    frame = gsd.hoomd.Frame()

    # forces declaration
    chain_force = harmonic
    chain_params = harmonic_params

    contact_force = harmonic2
    contact_params = harmonic2_params

    for bond_type in bond_types:
        chain_force.params[bond_type] = chain_params
        contact_force.params[bond_type] = contact_params

    colision_force = gaussian
    colision_force_params = gaussian_params
    
    for chs in itertools.combinations_with_replacement(bins.chrom.unique(), 2):
        bond_target = (chs[0], chs[1])
        colision_force.params[bond_target] = colision_force_params
        
    forces = [chain_force, contact_force, colision_force]

    # particles declaration
    frame.particles.N = particles_count
    frame.particles.position = particles_positions
    frame.particles.types = list(particles_chroms.unique())
    frame.particles.typeid = particles_id
    frame.configuration.box = [box_size, box_size, box_size, 0, 0, 0]

    # contacts declaration
    contact_count = np.count_nonzero(matrix)
    
    if debug == True:
        print('contact_count:', contact_count)
        print('len(bins.chrom):', len(frame.particles.types))

        print('particles:')
        print('num ', frame.particles.N)
        print('postitions num ', len(frame.particles.position))
        print('types ', frame.particles.types)
        print('types num ', len(frame.particles.typeid))

    bond_types.append('contact')
    chain_force.params['contact'] = dict(k = 0.0, r0 = 0)
    contact_force.params['contact'] = contact_params

    bonds_type_id = []
    bond_groups = []
    
    for i in range(1, particles_count):
        this_particle_type = particles_id[i]
        prev_particle_type = particles_id[i-1]

        if this_particle_type == prev_particle_type:
            bonds_type_id.append(this_particle_type)
            bond_groups.append([i-1, i])
            
    print('bond_groups num', len(bond_groups))
    print('bonds_type_id num', len(bonds_type_id))

    non_chain_contact_count = 0
    for (i,j),value in np.ndenumerate(matrix):
        if value != 0 and i != j:
            bond_groups.append([i, j])
            non_chain_contact_count+=1
    bonds_type_id.extend([len(bond_types)-1]*non_chain_contact_count)

    # bonds declaration
    frame.bonds.N = (frame.particles.N - 1) - (len(frame.particles.types) - 1) + non_chain_contact_count
    frame.bonds.types = bond_types
    frame.bonds.typeid = bonds_type_id
    frame.bonds.group = list(bond_groups)

    if debug == True:
        print('bonds:')
        print('num ', frame.bonds.N)
        print('types ', frame.bonds.types)
        print('typeid', frame.bonds.typeid[:10])
        print('chromosom_legend', chrom_name_to_id)
        print('unique types num', list(set(frame.bonds.typeid)))
        print('types num ', len(frame.bonds.typeid))
        print('groups num ', len(frame.bonds.group))

    # frame write
    if debug == False:
        with gsd.hoomd.open(name=frame_name, mode='x') as f:
            f.append(frame)
            
    return forces

def run_sim(state_dir: str, 
            device: hoomd.device,
            forces: list[Union[hoomd.md.bond.Harmonic, hoomd.md.pair.Gaussian]], 
            bins: pd.Series,
            no_traj_record: Optional[bool] = True,
            simple: Optional[bool] = False) -> None:
    """
    Runs simulation. After loading initiatory frame from .gsd file specified in state_dir will create hoomd.Simulation from it and add forces within hoomd.md.Integrator. Uses langevin model with kT=1.0 and time step of 0.005.
    
    Parameters
    ----------
    state_dir : str
        Path to the file holding initiated before hand first frame of the simulation.
    forces : list[hoomd.md.bond.Harmonic | hoomd.md.pair.Gaussian]
        List of forces to be added to the simulation.
    bins : pd.Series
        Bins description from the resolution to be run
    no_traj_record : Optional[bool] = True
        By defult only last frame of simulation will be saved. When set to False will save to the .gsd file full simulation record.
    simple : Optional[bool] = False) -> None
        When set to True will perform simulation without forces manipulations.
            
    Returns
    -------
    None
    
    Examples
    --------
    >>> run_sim('./test_frame.gsd', forces)
    """

    chain_force = forces[0]
    contact_force = forces[1]
    colision_force = forces[2]
    
    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_gsd(filename=state_dir)

    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    
    integrator = hoomd.md.Integrator(dt=0.001,
                                    methods=[langevin],
                                    forces=[chain_force, contact_force, colision_force])
    sim.operations.integrator = integrator

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    sim.operations.computes.append(thermodynamic_properties)

    if no_traj_record == False:
        write_period = 1
    else: 
        write_period = 8e4
        
    split = state_dir.rsplit('.', 1)
    traj_dir = split[0] + '_traj.' + split[1]
    gsd_writer = hoomd.write.GSD(filename=traj_dir,
                                trigger=hoomd.trigger.Periodic(int(write_period)),
                                mode='xb')
    sim.operations.writers.append(gsd_writer)

    logger = hoomd.logging.Logger()
    logger.add(chain_force, quantities=['energies', 'forces'])
    logger.add(contact_force, quantities=['energies', 'forces'])
    logger.add(colision_force, quantities=['energies', 'forces'])
    logger.add(thermodynamic_properties)
    gsd_writer.logger = logger
    
    
    if simple == False:
        # stage 1 no colision force
        for chs in itertools.combinations_with_replacement(bins.chrom.unique(), 2):
            bond_target = (chs[0], chs[1])
            colision_force.params[bond_target] = dict(epsilon = 0.0, sigma = 1)    
        sim.run(2e4)
        
        # stage 1 very weak colision force
        for chs in itertools.combinations_with_replacement(bins.chrom.unique(), 2):
            bond_target = (chs[0], chs[1])
            colision_force.params[bond_target] = dict(epsilon = 20.0, sigma = 0.5)    
        sim.run(2e4)
        
        # stage 1 weak colision force
        for chs in itertools.combinations_with_replacement(bins.chrom.unique(), 2):
            bond_target = (chs[0], chs[1])
            colision_force.params[bond_target] = dict(epsilon = 40.0, sigma = 1)    
        sim.run(2e4)
        
        # stage 4 full colision force
        for chs in itertools.combinations_with_replacement(bins.chrom.unique(), 2):
            bond_target = (chs[0], chs[1])
            colision_force.params[bond_target] = dict(epsilon = 100.0, sigma = 1)    
        sim.run(2e4)
    else:
        sim.run(8e4)
    
    gsd_writer.flush()
    
def perform_single_reconstruction(dir_str: str, 
                                  device: hoomd.device,
                                  hics_scales: list[np.ndarray], 
                                  bins_scales: list[pd.Series],
                                  stop_early: Optional[int] = 0, 
                                  save_each_iteration: Optional[bool] = False,
                                  save_screenshots: Optional[bool] = False, 
                                  visualize_result: Optional[bool] = False) -> None:
    """
    Performs single round of reconstruction.

    Parameters
    ----------
    dir_str : str
        Directory in which simulations records will be saved
    hics_scales : list[np.ndarray]
        List of HiC matrices used in each iteration
    bins_scales : list[pd.Series]
        List of bins descriptions used in each iteration
    stop_early : Optional[int] = 0
        Limits number of iterations stopping reconstruction earlier.
    save_each_iteration : Optional[bool] = False
        Specifies weather to save all iterations .gsd file or only final one. 
    save_screenshots : Optional[bool] = False
        Specifies weather to save images of each iteration resoult.
    visualize_result : Optional[bool] = False
        Specifies weather to visualize last iteration result at the end of reconstruction.

    Returns
    -------
    None

    Examples
    --------
    >>> perform_single_reconstruction('test5/', series_hics_scales[3], series_bins_scales[3], stop_early = 1)
    running iteration: 4
    bins num 42
    hic shape (42, 42)
    running sim...
    
    running iteration: 3
    bins num 56
    hic shape (56, 56)
    running sim...
    
    running iteration: 2
    bins num 84
    hic shape (84, 84)
    running sim...
    
    running iteration: 1
    bins num 169
    hic shape (169, 169)
    running sim...
    visualising results...
    """
    frame_str = 'frame_'
    gsd_str = '.gsd'
    trajectory_str = '_traj'
    iteration_num = len(hics_scales)

    for i in range(iteration_num-1, -1+stop_early, -1):
        this_frame_name = frame_str + str(i) + gsd_str
        this_traj_name = frame_str + str(i) + trajectory_str + gsd_str
        prev_traj_name = frame_str + str(i+1) + trajectory_str + gsd_str

        print()
        print('running iteration:', i)
        hics = hics_scales[i]
        bins = bins_scales[i]
        print('bins num', len(bins))
        print('hic shape', hics.shape)

        if i == (iteration_num-1):
            print('randomizing first frame...')
            forces = frame_initiation(hics, bins, dir_str+this_frame_name)
            # visualize_sim(dir_str+this_frame_name, no_visualize=True)
        else:
            print('generating new particles...')
            new_particles_position = generate_new_particles(dir_str+prev_traj_name, bins)
            print('initializing next frame...')
            forces = frame_initiation(hics, bins, dir_str+this_frame_name, new_particles_position)
            # visualize_sim(dir_str+this_frame_name, no_visualize=True)
            
        print('running sim...')
        run_sim(dir_str+this_frame_name, device, forces, bins)
        
        if i > 0 and save_screenshots == True:
            rev.visualize_sim(dir_str+this_traj_name, no_visualize=True)
        
        if i == 0+stop_early and visualize_result == True:
            print('visualising results...')
            rev.visualize_sim(dir_str+this_traj_name, screenshot=True)
            
        if i == 0+stop_early and save_each_iteration == False:
            for t in range(iteration_num-1, stop_early, -1):
                this_frame_name = frame_str + str(t) + gsd_str
                this_traj_name = frame_str + str(t) + trajectory_str + gsd_str
                os.remove(dir_str+this_traj_name)
                os.remove(dir_str+this_frame_name)
            os.remove(dir_str+frame_str+str(0+stop_early)+gsd_str)

def perform_many_reconstructions(dir_str: str, 
                                 device: hoomd.device,
                                 hics_scales: list[np.ndarray], 
                                 bins_scales: list[pd.Series],
                                 runs_num: int,
                                 stop_early: Optional[int] = 0,
                                 logs: Optional[str] = 'runs') -> None:
    """
    Performs many runs of the same reconstruction.

    Parameters
    ----------
    dir_str : str
        Directory in which simulations records will be saved
    hics_scales : list[np.ndarray]
        List of HiC matrices used in each iteration
    bins_scales : list[pd.Series]
        List of bins descriptions used in each iteration
    runs_num : int
        Controls number of runs
    stop_early : Optional[int] = 0
        Limits number of iterations stopping reconstruction earlier.
    logs : Optional[str] = 'runs'
        Set to 'runs' or 'iterations' to control printing of steps taking place. Default value is 'runs'.

    Returns
    -------
    None

    Examples
    --------
    >>> perform_many_reconstructions('test6/', series_hics_scales[2], series_bins_scales[2], runs_num = 5, stop_early = 1)
    running run 0
    running iteration: 4
    bins num 42
    hic shape (42, 42)
    running iteration: 3
    bins num 56
    hic shape (56, 56)
    running iteration: 2
    bins num 84
    hic shape (84, 84)
    running iteration: 1
    bins num 169
    hic shape (169, 169)

    running run 1
    running iteration: 4
    bins num 42
    hic shape (42, 42)
    running iteration: 3
    bins num 56
    hic shape (56, 56)
    running iteration: 2
    bins num 84
    hic shape (84, 84)
    ...
    hic shape (84, 84)
    running iteration: 1
    bins num 169
    hic shape (169, 169)
    """
    run_str = 'run_'
    frame_str = '_frame_'
    gsd_str = '.gsd'
    trajectory_str = '_traj'
    iteration_num = len(hics_scales)
    
    for j in range(runs_num):
        
        if logs == 'runs' or logs == 'iterations':
            print()
            print('running run', j)

        for i in range(iteration_num-1, -1+stop_early, -1):
            this_frame_name = run_str + str(j) + frame_str + str(i) + gsd_str
            this_traj_name = run_str + str(j) + frame_str + str(i) + trajectory_str + gsd_str
            prev_traj_name = run_str + str(j) + frame_str + str(i+1) + trajectory_str + gsd_str

            hics = hics_scales[i]
            bins = bins_scales[i]
            if logs == 'iterations':
                print('running iteration:', i)
                print('bins num', len(bins))
                print('hic shape', hics.shape)

            if i == (iteration_num-1):
                forces = frame_initiation(hics, bins, dir_str+this_frame_name)
            else:
                new_particles_position = generate_new_particles(dir_str+prev_traj_name, bins)
                forces = frame_initiation(hics, bins, dir_str+this_frame_name, new_particles_position)

            run_sim(dir_str+this_frame_name, device, forces, bins) 
        
            if i == 0+stop_early:
                for t in range(iteration_num-1, stop_early, -1):
                    this_frame_name = run_str + str(j) + frame_str + str(t) + gsd_str
                    this_traj_name = run_str + str(j) + frame_str + str(t) + trajectory_str + gsd_str
                    os.remove(dir_str+this_traj_name)
                    os.remove(dir_str+this_frame_name)
                os.remove(dir_str+run_str+str(j)+frame_str+str(0+stop_early)+gsd_str)