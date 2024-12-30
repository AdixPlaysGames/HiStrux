import numpy as np
import gsd
import matplotlib as mpl
import pyvista as pv

def visualize_sim(trajectory_dir, no_contacts=True, debug=False, screenshot=False):
    # trajectory read
    with gsd.hoomd.open(trajectory_dir) as f:
        frame = f[-1]

    # frame data read
    particles_position = frame.particles.position
    particles_types = frame.particles.types
    particles_types_id = frame.particles.typeid

    chromosom_legend = dict(zip(np.unique(particles_types_id), particles_types))
    particles_chrom = [chromosom_legend.get(item, item) for item in particles_types_id]

    bonds_particles = frame.bonds.group
    bonds_types = frame.particles.types + ['contact']
    bonds_types_id = frame.bonds.typeid
    bonds_legend = dict(zip(np.unique(bonds_types_id), bonds_types))
    bonds_chrom = [bonds_legend.get(item, item) for item in bonds_types_id]
    
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
        print('bonds_chrom:', bonds_chrom[:5])

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
    plotter = pv.Plotter()

    # add particles
    point_cloud = pv.PolyData(particles_position)
    point_cloud['chromosom'] = particles_chrom

    plotter.add_mesh(
        point_cloud, 
        scalars = 'chromosom',
        cmap = my_cmap_chroms,
        point_size = 10, 
        render_points_as_spheres = True)
    
    # add chain bonds
    lines = pv.MultiBlock()

    for b in range(len(bonds_particles_no_contact)):
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
        line_width = 5
    )

    # add contacts bonds
    if no_contacts == False:
        contact_lines = pv.MultiBlock()

        for b in range(len(bonds_particles_contact)):
            start_particle = bonds_particles_contact[b][0]
            end_particle = bonds_particles_contact[b][1]

            start_position = particles_position[start_particle]
            end_position = particles_position[end_particle]

            line = pv.Line(start_position, end_position)
            line.point_data['chromosom'] = 'contact'#[bonds_chrom_contact[b]]*2
            # print(line.point_data['chromosom'])
            contact_lines[str(b)] = line

        merged_contact_lines = contact_lines.combine()

        plotter.add_mesh(
            merged_contact_lines,
            scalars = 'chromosom',
            cmap = my_cmap_contacts,
            line_width = 0.25
        )

    if screenshot:
        plotter.show()
    else:
        plotter.show(screenshot=trajectory_dir.rsplit('.', 1)[0])
        
def inspect_gsd(gsd_traj):
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
    bonds_chrom = [bonds_legend.get(item, item) for item in bonds_types_id]

    print('chromosom_legend:', chromosom_legend)
    print('particles_position:', particles_position[:5])
    print('particles_types:', particles_types)
    print('particles_types_id:', particles_types_id[:5])
    print('particles_chrom:', particles_chrom)
    print('bonds_particles:', bonds_particles[:5])
    print('bonds_types:', bonds_types)
    print('bonds_types_id:', bonds_types_id[:5])
    print('bonds_legend:', bonds_legend)
    print('bonds_chrom:', bonds_chrom[:5])