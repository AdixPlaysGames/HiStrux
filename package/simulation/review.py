import numpy as np
import gsd
import matplotlib as mpl
import pyvista as pv
from typing import Optional, Union

def remove_contact_bonds(gsd_trjectory: str) -> None:
	"""
	Will remove contact bonds from all frames of the simulation record .gsd file. Usefull for visualizations using 3rd party software.

	Parameters
	----------
	gsd_trajectory : str
		Path to the .gsd file containing simulation record.
	Returns
	-------
	None

	Examples:
	>>>	remove_contact_bonds('./test7/frame_4_traj.gsd')
	File saved to the ./test7/frame_4_traj_no_contact.gsd
	>>>	remove_contact_bonds('./test7/frame_4_traj.gsd')
	An error occurred: [Errno 17] File exists: './test7/frame_4_traj_no_contact.gsd'
	"""
	new_frames = []

	with gsd.hoomd.open(gsd_trjectory) as f:
		frames = f[:]
		for i in range(len(frames)):
			frame = frames[i]

			bonds_group = frame.bonds.group
			bonds_types = frame.bonds.types
			bonds_types_id = frame.bonds.typeid

			no_contact_idx = bonds_types_id != 5

			frame.bonds.group = bonds_group[no_contact_idx]
			frame.bonds.types = bonds_types[:-1]
			frame.bonds.typeid = bonds_types_id[no_contact_idx]
			frame.bonds.N = len(frame.bonds.typeid)
			new_frames.append(frame)

	
	parts = gsd_trjectory.rsplit('.', 1)
	gsd_trjectory_no_contact = parts[0] + '_no_contact.' + parts[1]

	try: 
		with gsd.hoomd.open(gsd_trjectory_no_contact, mode='x') as f:
			f.extend(new_frames)
		print('File saved to the '+gsd_trjectory_no_contact)
	except Exception as e:
		print("An error occurred:", str(e))

def visualize_sim(trajectory_dir: str, 
                  no_contacts: Optional[bool] = True, 
                  screenshot: Optional[bool] = False, 
                  no_visualize: Optional[bool] = False, 
                  chroms: Optional[list[str]] = None, 
                  chain_width: Optional[int] = 5, 
                  contact_width: Optional[int] = 0.25, 
                  particle_size: Optional[int] = 10,
                  debug: Optional[bool] = False) -> None:
    """
    Visualizes simulation end state from last frame of the trajectory_dir .gsd file specified. Uses pyvista package and allows for selection of chromosomes, visualization settings and whether to save its picture.
    
    Parameters
    ----------
    trajectory_dir : str, 
        Path to the gsd file holding simulation state to be visualized.
    no_contacts: Optional[bool] = True
        When set to False will also show lines representing contact forces acting between particles. For bigger simulation wil cosiderably lower vision of strucutre.
    screenshot : Optional[bool] = False
        When set to True will additionaly save generated visualization.
    no_visualize : Optional[bool] = False
        When set to True will only save generated visualization without showing it.
    chroms : Optional[list[str]] = None
        Specifies which chromosomes will be visualized.
    chain_width : Optional[int] = 5
        Specifies width of lines representing chain forces acting between particles.
    contact_width : Optional[int] = 0.25
        Specifies width of lines representing contact forces acting between particles.
    particle_size : Optional[int] = 10
        Specifies size spheres representing particles.
    debug: Optional[bool] = False
        Prints control info during execution. 

    Returns
    -------
    None

    Examples
    --------
    >>> visualize_sim('./test_frame_traj.gsd', screenshot=True)
    """
    
    # trajectory read
    with gsd.hoomd.open(trajectory_dir) as f:
        frame = f[-1]

    # frame data read
    particles_position = frame.particles.position
    particles_types = frame.particles.types
    particles_types_id = frame.particles.typeid

    bonds_particles = frame.bonds.group
    bonds_types = frame.particles.types + ['contact']
    bonds_types_id = frame.bonds.typeid
        
    chromosom_legend = dict(zip(np.unique(particles_types_id), particles_types))
    particles_chrom = [chromosom_legend.get(item, item) for item in particles_types_id]
    bonds_legend = dict(zip(np.unique(bonds_types_id), bonds_types))
    bonds_legen_test = {i: bond for i, bond in enumerate(bonds_types)}
    bonds_chrom = [bonds_legen_test.get(item, item) for item in bonds_types_id] #test
    
    if debug == True:
        print('chromosom_legend:', chromosom_legend)
        print('particles_position:', particles_position[:5])
        print('particles_types:', particles_types)
        print('particles_types_id:', particles_types_id[:5])
        print('particles_chrom:', particles_chrom)
        print('bonds_particles:', bonds_particles[:5])
        print('bonds_types:', bonds_types)
        print('bonds_types_id:', bonds_types_id[:5])
        print('bonds_legend:', bonds_legend)
        print('bonds_legen_test', bonds_legen_test)
        print('bonds_chrom:', bonds_chrom[:])

    # set color mapping 
    og_cmap = mpl.colormaps['tab20']
    my_colors = og_cmap(np.linspace(0, 1, len(particles_types)+1))
    my_cmap_chroms = mpl.colors.ListedColormap(my_colors[:-1])
    my_cmap_contacts = mpl.colors.ListedColormap(my_colors[-1])

    # separate chain bonds from contacts
    no_contact_check = np.isin(bonds_chrom, ['contact'])
    bonds_particles_no_contact = []
    bonds_chrom_no_contact = []
    bonds_particles_contact = []
    bonds_chrom_contact = []

    for i in range(len(bonds_particles)):
        if no_contact_check[i] == False:
            bonds_particles_no_contact.append(bonds_particles[i])
            bonds_chrom_no_contact.append(bonds_chrom[i])

        if no_contact_check[i] == True:
            bonds_particles_contact.append(bonds_particles[i])
            bonds_chrom_contact.append(bonds_chrom[i])

    bonds_particles_no_contact = np.vstack(bonds_particles_no_contact)
    bonds_chrom_no_contact = np.vstack(bonds_chrom_no_contact)
    bonds_particles_contact = np.vstack(bonds_particles_contact)
    bonds_chrom_contact = np.vstack(bonds_chrom_contact)

    bonds_chrom_no_contact = bonds_chrom_no_contact.ravel().tolist()
    bonds_chrom_contact = bonds_chrom_contact.ravel().tolist()

    if debug == True:
        print('bonds_particles_no_contact', bonds_particles_no_contact[-5:])
        print('bonds_chrom_no_contact', bonds_chrom_no_contact[-5:])
        print('bonds_particles_contact', bonds_particles_contact[-5:])
        print('bonds_chrom_contact', bonds_chrom_contact[-5:])

    # define plot
    if no_visualize:
        plotter = pv.Plotter(off_screen=True)
    else:
        plotter = pv.Plotter()

    # add particles
    if chroms != None:
        concat_particles_info = zip(particles_position, particles_types_id)
        filtered_particles_info = [(position, type_id) for position, type_id in concat_particles_info if chromosom_legend[type_id] in chroms + ['contact']]
        filtered_particles_position, filtered_particles_types_id = zip(*filtered_particles_info)
        filtered_particles_position = list(filtered_particles_position)
        filtered_particles_types_id = list(filtered_particles_types_id)
        filtered_particles_chrom = [chromosom_legend.get(item, item) for item in filtered_particles_types_id]

        point_cloud = pv.PolyData(filtered_particles_position)
        point_cloud['chromosom'] = filtered_particles_chrom
    else:
        point_cloud = pv.PolyData(particles_position)
        point_cloud['chromosom'] = particles_chrom

    plotter.add_mesh(
        point_cloud, 
        scalars = 'chromosom',
        cmap = my_cmap_chroms,
        point_size = particle_size, 
        render_points_as_spheres = True)
    
    # add chain bonds
    lines = pv.MultiBlock()

    for b in range(len(bonds_particles_no_contact)):
        # if (chroms != None and bonds_legend[bonds_types_id[b]] in chroms) or chroms == None:
        if (chroms != None and bonds_legen_test[bonds_types_id[b]] in chroms) or chroms == None:
            start_particle = bonds_particles_no_contact[b][0]
            end_particle = bonds_particles_no_contact[b][1]

            start_position = particles_position[start_particle]
            end_position = particles_position[end_particle]

            line = pv.Line(start_position, end_position)
            line.point_data['chromosom'] = [bonds_chrom_no_contact[b]]*2
            lines[str(b)] = line

    merged_lines = lines.combine()

    plotter.add_mesh(
        merged_lines,
        scalars = 'chromosom',
        cmap = my_cmap_chroms,
        line_width = chain_width
    )

    # add contacts bonds
    if no_contacts == False:
        contact_lines = pv.MultiBlock()

        for c in range(len(bonds_particles_contact)):
            start_particle = bonds_particles_contact[c][0]
            end_particle = bonds_particles_contact[c][1]

            start_position = particles_position[start_particle]
            end_position = particles_position[end_particle]

            line = pv.Line(start_position, end_position)
            line.point_data['chromosom'] = 'contact'#[bonds_chrom_contact[b]]*2
            # print(line.point_data['chromosom'])
            contact_lines[str(c)] = line

        merged_contact_lines = contact_lines.combine()

        plotter.add_mesh(
            merged_contact_lines,
            scalars = 'chromosom',
            cmap = my_cmap_contacts,
            line_width = contact_width
        )

    if no_visualize == True:
        plotter.screenshot(trajectory_dir.rsplit('.', 1)[0])
        print('screenshot, no visualization')
    else:
        if screenshot == False:
            plotter.show()
            print('no screenshot, visualization')
        else:
            plotter.show(screenshot=trajectory_dir.rsplit('.', 1)[0])
            print('screenshot, visualization')
        
def inspect_gsd(gsd_traj: str) -> None:
    """
    Prints out information about particles from last simulation frame within specified .gsd file.
    
    Parameters
    ----------
    gsd_traj : str
        Directory of a .gsd file to be inspected.

    Returns
    -------
    None

    Examples
    --------
    >>> inspect_gsd('./test_frame_traj.gsd')
    chromosom_legend: {0: 'chr1', 1: 'chr2', 2: 'chr3', 3: 'chr4', 4: 'chr5'}
	particles num 169
	particles_position: [[ -9.152195   14.470503    4.625657 ]
	[ -8.966829   13.731618    4.279521 ]
	[ -9.085836   14.734003    3.9680572]
	[-10.579327   13.710347    4.338241 ]
	[-10.103192   13.565733    4.786075 ]]
	particles_types: ['chr1', 'chr2', 'chr3', 'chr4', 'chr5']
	particles_types_id: [0 0 0 0 0]
	particles_chrom: ['chr1', 'chr1', 'chr1', 'chr1', 'chr1']

	bonds num 5018
	bonds_particles: [[0 1]
	[1 2]
	[2 3]
	[3 4]
	[4 5]]
	bonds_types: ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'contact']
	bonds_types_id: [0 0 0 0 0]
	bonds_legend: {0: 'chr1', 1: 'chr2', 2: 'chr3', 3: 'chr4', 4: 'chr5', 5: 'contact'}
	bonds_legen_test {0: 'chr1', 1: 'chr2', 2: 'chr3', 3: 'chr4', 4: 'chr5', 5: 'contact'}
	bonds_chrom: ['chr1', 'chr1', 'chr1', 'chr1', 'chr1']
	no_contact_check [False False False ...  True  True  True]
	bonds_particles_no_contact [array([0, 1], dtype=uint32), array([1, 2], dtype=uint32), array([2, 3], dtype=uint32), array([3, 4], dtype=uint32), array([4, 5], dtype=uint32)]
	bonds_chrom_no_contact ['chr1', 'chr1', 'chr1', 'chr1', 'chr1']
	bonds_particles_contact [array([0, 2], dtype=uint32), array([0, 3], dtype=uint32), array([0, 4], dtype=uint32), array([0, 5], dtype=uint32), array([0, 6], dtype=uint32)]
	bonds_chrom_contact ['contact', 'contact', 'contact', 'contact', 'contact']
    """
    
    with gsd.hoomd.open(gsd_traj) as f:
        frame = f[-1]
    
    particles_position = frame.particles.position
    particles_types = frame.particles.types
    particles_types_id = frame.particles.typeid

    chromosom_legend = dict(zip(np.unique(particles_types_id), particles_types))
    particles_chrom = [chromosom_legend.get(item, item) for item in particles_types_id]

    bonds_particles = frame.bonds.group
    bonds_types = frame.particles.types + ['contact']
    bonds_types_id = frame.bonds.typeid
    bonds_legend = dict(zip(np.unique(bonds_types_id), bonds_types))
    bonds_legen_test = {i: bond for i, bond in enumerate(bonds_types)}
    bonds_chrom = [bonds_legen_test.get(item, item) for item in bonds_types_id]

    print('chromosom_legend:', chromosom_legend)
    print('particles num', len(particles_types_id))
    print('particles_position:', particles_position[:5])
    print('particles_types:', particles_types)
    print('particles_types_id:', particles_types_id[:5])
    print('particles_chrom:', particles_chrom[:5])
    print()
    print('bonds num', len(bonds_types_id))
    print('bonds_particles:', bonds_particles[:5])
    print('bonds_types:', bonds_types)
    print('bonds_types_id:', bonds_types_id[:5])
    print('bonds_legend:', bonds_legend)
    print('bonds_legen_test', bonds_legen_test)
    print('bonds_chrom:', bonds_chrom[:5])
    
    no_contact_check = np.isin(bonds_chrom, ['contact'])
    bonds_particles_no_contact = []
    bonds_chrom_no_contact = []
    bonds_particles_contact = []
    bonds_chrom_contact = []

    print('no_contact_check', no_contact_check)
    
    for i in range(len(bonds_particles)):
        if no_contact_check[i] == False:
            bonds_particles_no_contact.append(bonds_particles[i])
            bonds_chrom_no_contact.append(bonds_chrom[i])

        if no_contact_check[i] == True:
            bonds_particles_contact.append(bonds_particles[i])
            bonds_chrom_contact.append(bonds_chrom[i])

    print('bonds_particles_no_contact', bonds_particles_no_contact[:5])
    print('bonds_chrom_no_contact', bonds_chrom_no_contact[:5])
    print('bonds_particles_contact', bonds_particles_contact[:5])
    print('bonds_chrom_contact', bonds_chrom_contact[:5])

    bonds_particles_no_contact = np.vstack(bonds_particles_no_contact)
    bonds_chrom_no_contact = np.vstack(bonds_chrom_no_contact)
    bonds_particles_contact = np.vstack(bonds_particles_contact)
    bonds_chrom_contact = np.vstack(bonds_chrom_contact)
