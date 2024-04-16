import energyflow as ef
import numpy as np
from pyjet import cluster


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


def center_sort_and_normalize(events, normalize = True):
    """Function to center and normalize a dataset of events
    Args:
        events (ndarray): Dataset of events with shape (n_events, n_particles, 3)
        normalize (bool, optional): Whether to normalize the events to 1. Defaults to True.

    Returns:
        ndarray: Centered and normalized dataset of events
    """


    centered_events = []
    for event in events:
        # Center Jets
        mask = event[:, 0] > 0
        yphi_avg = np.average(event[mask, 1:], weights=event[mask, 0], axis=0)
        event[mask, 1:] -= yphi_avg

        # Sort by pT
        indices = (-event[:, 0]).argsort()
        event = event[indices]
        

        # Normalize
        if normalize:
            norm = np.sum(event[mask, 0]) if normalize else 1
            event[mask, 0] = event[mask, 0] / norm
            

        centered_events.append(event)

    return np.array(centered_events)



def load_cmsopendata(cache_dir, dataset, pt_lower, pt_upper, eta, quality, normalize_pad = True, return_kfactors=True, pad = 125, momentum_scale=1, n=1000, amount=1):
    """Function to load CMS Open Data from the energyflow package
    
    Args:
        cache_dir (str): Directory to cache the data
        dataset (str): Dataset to load. Either "cms", "sim", or "gen"
        pt_lower (float): Lower bound on the jet $p_T$
        pt_upper (float): Upper bound on the jet $p_T$
        eta (float): Upper bound on the jet |$\eta$|
        quality (int): Quality of the jet
        normalize_pad (bool, optional): Whether to normalize the events to 1 $after$ padding. Defaults to True.
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
    n = min(n, sim.weights.shape[0] )
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
            norm = np.sum(zs[:pad]) if normalize_pad else 1
            events[i, :, 0] = zs[:pad] / norm
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
    

def random_triangles(n_samples, R, phase_space = True):
    """Function to generate a dataset of N random triangles with radius R

    Args:
        N (int): Number of triangles to generate
        R (float): Radius of the triangle
        phase_space (bool, optional): Whether to generate the triangles using 3-body Lorentz invariant phase space, defaults to True.

    Returns:
        ndarray: Dataset of triangles with shape (N, 3, 3)
    """

    n_points = 3
    if not phase_space:
        dataset = np.zeros((n_samples, n_points, 3))

        for i in range(n_samples):  

            # Randomly generate 3 points in the unit disk
            zs = np.random.uniform(0, 1, n_points)
            zs = zs / np.sum(zs)
            rad = np.random.uniform(0, 1, n_points)
            theta = np.random.uniform(0, 2*np.pi, n_points)

            # sqrts for appropriate density
            x = np.sqrt(rad) * R * np.cos(theta)
            y = np.sqrt(rad) * R * np.sin(theta)

            dataset[i, :, 0] = zs
            dataset[i, :, 1] = x
            dataset[i, :, 2] = y

        weights = np.ones(n_samples)

        return dataset, weights
    
    else:

        # Generate according to 3-body phase space
        # dPhi3 ~ d(theta12)^2 d(theta13)^2 d(phi) d(z1) d(z2) * z1 * z2 * z3

        dataset = np.zeros((n_samples, n_points, 3))
        weights = np.zeros(n_samples)

        for i in range(n_samples):
            
            # Generate z1, z2, z3 with accept/reject such that z1 + z2 + z3 = 1 
            z3 = -1
            while(z3 < 0):
                z1 = np.random.uniform(0, 1)
                z2 = np.random.uniform(0, 1)
                z3 = 1 - z1 - z2
            dataset[i, 0, 0] = z1
            dataset[i, 1, 0] = z2
            dataset[i, 2, 0] = z3
            weights[i] = z1 * z2 * z3

            # Generate theta12, theta13, phi
            theta12_squared = np.random.uniform(0, R**2)
            theta13_squared = np.random.uniform(0, R**2)
            phi = np.random.uniform(0, 2*np.pi)
            overall_phase = np.random.uniform(0, 2*np.pi)

            # calculate x and y coordinates
            x1 = 0
            y1 = 0
            x2 = np.sqrt(theta12_squared) * np.cos(overall_phase)
            y2 = np.sqrt(theta12_squared) * np.sin(overall_phase)
            x3 = np.sqrt(theta13_squared) * np.cos(phi + overall_phase)
            y3 = np.sqrt(theta13_squared) * np.sin(phi + overall_phase)

            dataset[i, 0, 1] = x1
            dataset[i, 0, 2] = y1
            dataset[i, 1, 1] = x2
            dataset[i, 1, 2] = y2
            dataset[i, 2, 1] = x3
            dataset[i, 2, 2] = y3

        return dataset, weights
    

def kT_N(events, N, R):

    jets = []

    for event in events:

        # Set up 4-vectors
        four_vectors = []
        for particle in event:
            four_vectors.append((particle[0], particle[1], particle[2], 0))
        four_vectors = np.array(four_vectors, dtype=[("pt", "f8"), ("eta", "f8"), ("phi", "f8"), ("mass", "f8")])

        # Cluster with kT (p = 1)
        sequence = cluster(four_vectors, R=R, p=1)
        subjets = sequence.exclusive_jets(N)

        output = np.zeros((N, 3))
        for i, subjet in enumerate(subjets):
            output[i,0] = subjet.pt
            output[i,1] = subjet.eta
            output[i,2] = subjet.phi


        # Normalize
        output[:,0] = np.nan_to_num(output[:,0] / np.sum(output[:,0]))

        jets.append(output)


    return np.array(jets)