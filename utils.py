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


def load_cmsopendata(cache_dir, dataset, pt_lower, pt_upper, eta, quality, return_kfactors=True, pad = 125, momentum_scale=1, n=1000, amount=1, frac=1.0):

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
            events[i, 0, :] = zs[:pad]
            events[i, 1:, :] = points[:pad]


    particle_counts = np.array(particle_counts)

    print("Max # of particles: %d" % max(particle_counts))

    # kfactors
    if return_kfactors and dataset == "sim":
        print("test")
        return events, weights, ef.mod.kfactors('sim', sim.corr_jet_pts, sim.npvs)
    elif return_kfactors and dataset == "gen":
        return events, weights, ef.mod.kfactors('gen', sim.jet_pts,)

    return events, weights