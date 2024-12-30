import numpy as np
import gsd.hoomd
import hoomd
import itertools

def generate_new_particles(trajectory_dir, end_num, debug=False):
    with gsd.hoomd.open(trajectory_dir) as f:
        frame = f[-1]
    particles_position = frame.particles.position
    
    bonds_position_start = particles_position[:-1]
    bonds_position_end = particles_position[1:]
    # bonds_position = np.ndarray([end_num, 2, 3])
    # bonds_position[:, 0, :] = bonds_position_start
    # bonds_position[:, 1, :] = bonds_position_end

    bonds_length = np.linalg.norm(bonds_position_end - bonds_position_start, axis=1)
    bonds_cumsum = np.cumsum(bonds_length)
    new_dist = sum(bonds_length) / (end_num - 1)
    new_bonds_cumsum = np.cumsum([0] + [new_dist]*(end_num - 2))
    new_idxs = np.searchsorted(bonds_cumsum, new_bonds_cumsum)

    if debug == True:
        print(bonds_cumsum)
        print(new_bonds_cumsum[1:])
        print(new_idxs)

    bonds_dir = (bonds_position_end - bonds_position_start) / np.array([bonds_length]).T
    
    if debug == True:
        print(bonds_dir)
        print(particles_position[new_idxs])
        print(bonds_dir[new_idxs])

    segment_dist = np.hstack((np.array([0]), bonds_cumsum))
    new_segment_dist = new_bonds_cumsum 

    if debug == True:
        print('particles_position', particles_position[:5])
        print('bonds_dir', bonds_dir[:5])
        print('segment_dist', segment_dist[:5])
        print('new_segment_dist', new_segment_dist[:5])
        print('new_idxs', new_idxs[:5])

    new_particles_position = particles_position[new_idxs] + bonds_dir[new_idxs] * np.array([new_segment_dist - segment_dist[new_idxs]] * 3).T
    new_particles_position = np.vstack((new_particles_position, particles_position[-1]))

    return new_particles_position

def generate_initial_positions(particles_count, radius, box_size):
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

def frame_initiation(matrix, bins, frame_name, new_particles_position=None, debug=False):
    chrom_name_to_id = {element : i for i, element in enumerate(set(bins.chrom))}
    box_size = 350

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

    gaussian_params = dict(epsilon = 100.0, sigma = 0.1)
    gaussian_nlist_params = dict(buffer = 1, default_r_cut=3.0)
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
    colision_params = gaussian_params

    for chs in itertools.combinations_with_replacement(bins.chrom.unique(), 2):
        bond_target = (chs[0], chs[1])
        colision_force.params[bond_target] = colision_params
        
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
    contact_force.params['contact'] = dict(k = 2000.0, r0 = 1.5)

    bonds_type_id = []
    bond_groups = []
    
    for i in range(1, particles_count):
        this_particle_type = particles_id[i]
        prev_particle_type = particles_id[i-1]

        if this_particle_type == prev_particle_type:
            bonds_type_id.append(this_particle_type)
            bond_groups.append([i-1, i])

    bonds_type_id.extend([len(bond_types)-1]*contact_count)

    for (i,j),value in np.ndenumerate(matrix):
        if value != 0:
            bond_groups.append([i, j])

    # bonds declaration
    frame.bonds.N = (frame.particles.N - 1) - (len(frame.particles.types) - 1) + contact_count
    frame.bonds.types = bond_types
    frame.bonds.typeid = bonds_type_id
    frame.bonds.group = list(bond_groups)

    if debug == True:
        print('bonds:')
        print('num ', frame.bonds.N)
        print('types ', frame.bonds.types)
        print('types num ', len(frame.bonds.typeid))
        print('groups num ', len(frame.bonds.group))

    # frame write
    if debug == False:
        with gsd.hoomd.open(name=frame_name, mode='x') as f:
            f.append(frame)
            
    return forces

def run_sim(state_dir, forces, no_colision=False):
    chain_force = forces[0]
    contact_force = forces[1]
    colision_force = forces[2]

    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_gsd(filename=state_dir)

    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)

    if no_colision:
        integrator = hoomd.md.Integrator(dt=0.005,
                                        methods=[langevin],
                                        forces=[chain_force, contact_force])
    else:
        integrator = hoomd.md.Integrator(dt=0.005,
                                        methods=[langevin],
                                        forces=[chain_force, contact_force, colision_force])
    sim.operations.integrator = integrator

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    sim.operations.computes.append(thermodynamic_properties)

    split = state_dir.rsplit('.', 1)
    traj_dir = split[0] + '_traj.' + split[1]
    gsd_writer = hoomd.write.GSD(filename=traj_dir,
                                trigger=hoomd.trigger.Periodic(1),
                                mode='xb')
    sim.operations.writers.append(gsd_writer)

    logger = hoomd.logging.Logger()
    logger.add(chain_force, quantities=['energies', 'forces'])
    logger.add(contact_force, quantities=['energies', 'forces'])
    logger.add(colision_force, quantities=['energies', 'forces'])
    logger.add(thermodynamic_properties)
    gsd_writer.logger = logger
    
    sim.run(8e3)
    gsd_writer.flush()
    