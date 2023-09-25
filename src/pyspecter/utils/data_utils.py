import energyflow as ef
import numpy as np

# Energy-flow package for CMS Open Data loader
energyflow_flag = True
try:
    import energyflow as ef
except ImportError:
    energyflow_flag = False
    print("Warning: Package imageio not found. Cannot use gif creation functionality!")


# ###############################
# ########## LOAD DATA ##########
# ###############################


def load_cmsopendata(cache_dir, dataset, pt_lower, pt_upper, eta, quality, return_kfactors=True, pad = 125, momentum_scale=1, n=1000, amount=1):
    """Function to load CMS Open Data from the energyflow package
    
    Args:
        cache_dir (str): Directory to cache the data
        dataset (str): Dataset to load. Either "cms", "sim", or "gen"
        pt_lower (float): Lower bound on the jet $p_T$
        pt_upper (float): Upper bound on the jet $p_T$
        eta (float): Upper bound on the jet |$\eta$|
        quality (int): Quality of the jet
        return_kfactors (bool, optional): Whether to return the k-factor weights. Defaults to True.
        pad (int, optional): Number of particles to pad the events to. Defaults to 125.
        momentum_scale (float, optional): Common scale factor to divide all momenta by to keep parameters ~ 1. Defaults to 1.
        n (int, optional): Number of events to load. Defaults to 1000.
        amount (float, optional): Number of files to load at a time. Defaults to 1. If a float, the fraction of files to load. Note that 1.00 will load all files, while integer 1 will load only the first file.
    """

    if not energyflow_flag:
        raise ImportError('Need the energyflow package to use the default CMS data loader!')

    # Load data
    specs = [f'{pt_lower} <= corr_jet_pts <= {pt_upper}', f'abs_jet_eta < {eta}', f'quality >= {quality}']
    sim = ef.mod.load(*specs, cache_dir=cache_dir, dataset=dataset, amount=amount)


    # CMS JEC's
    C = sim.jets_f[:, sim.jec]

    # PFC's
    pfcs = sim.particles
    weights = sim.weights * sim.weights.shape[0] / n


    # PFC's
    events = np.zeros((n, pad, 3))
    particle_counts = []

    for (i, jet) in enumerate(pfcs[:n]):

        indices = (-jet[:, 0]).argsort()
        zs = jet[indices, 0] / np.sum(jet[indices, 0])
        points = jet[indices, 1:3]

        # Center Jets
        mask = zs > 0
        yphi_avg = np.average(points[mask], weights=zs[mask], axis=0)
        points[mask] -= yphi_avg

        num_particles = jet.shape[0]
        particle_counts.append(num_particles)

        # 0 padding for rectangular arrays
        if num_particles < pad:
            events[i, :num_particles, 0] = zs
            events[i, :num_particles, 1:] = points

        else:
            events[i, :, 0] = zs[:pad]
            events[i, ::, 1:] = points[:pad]


    particle_counts = np.array(particle_counts)

    print("Max # of particles: %d" % max(particle_counts))

    # kfactors
    if return_kfactors and dataset == "sim":
        return events, weights, ef.mod.kfactors('sim', sim.corr_jet_pts, sim.npvs)
    elif return_kfactors and dataset == "gen":
        return events, weights, ef.mod.kfactors('gen', sim.jet_pts,)

    return events, weights


def load_triangles(num_angles, num_energies, R, return_indices = False):
    """Function to load a dataset of isosceles triangles with a given number of angles (from 0 to $\pi$) and energies (from 0 to 1)

    Args:
        num_angles (int): Number of angles to use
        num_energies (int): Number of energies to use
        R (float): Radius of the triangle
        return_indices (bool, optional): Whether to return the indices of the dataset. Defaults to False.

    Returns:
        ndarray: Dataset of triangles with shape (num_angles*num_energies, 3, 3). Sorted by angle, then energy.
    """


    # Build 3 particle events
    pad = 3
    angles = np.linspace(0, np.pi, num_angles)
    energies = np.linspace(0, 1, num_energies)
    ij_indices = []

    events = []
    for i in range(num_angles):
        temp = []
        for j in range(num_energies):

            points = np.array([
                                (energies[j], 0.0, 0.0),
                                ((1-energies[j])/2, R, 0.0),
                                ((1-energies[j])/2, R*np.cos(angles[i]), R*np.sin(angles[i]))
                    ])
            temp.append(points)
            ij_indices.append((i,j))
        events.append(temp)


    ij_indices = np.array(ij_indices)
    dataset = np.array(events)[ij_indices[:,0], ij_indices[:,1]]

    if return_indices:
        return dataset, ij_indices
    else:
        return dataset