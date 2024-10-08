# Standard Imports
import numpy as np
from time import time
from matplotlib import pyplot as plt
import os

from particleloader import load

from pyspecter.SPECTER import SPECTER
from pyspecter.CommonObservables import build_event_observables
# from pyspecter.SpecialObservables import SpecialObservables



# %%%%%%%%%% Load Data %%%%%%%%%%

# Parameters 
R = 0.8
this_dir = "/n/home01/rikab/SPECTER_STUDIES"
this_study = "ee_dijets"
lr = 0.005
epochs = 500

batch_size = 10000


# Events
events_dir = "/n/holyscratch01/iaifi_lab/rikab/top/top_events.npy"
events_dir = "/n/holyscratch01/iaifi_lab/rikab/ee_dijets/lep_dijets.npy"
# data = np.load("/n/holyscratch01/iaifi_lab/rikab/ee_dijets/lep_dijets.npy", allow_pickle=True)

dataset_open = load("SPECTER_ee_dijets", 100000, "/n/holyscratch01/iaifi_lab/rikab/.ParticleLoader").astype(np.float64)[:,:128,:]

# for (i, x) in enumerate(dataset_open):


#     # theta = np.arccos(x[:,3] / x[:,0])
#     # eta = -np.log(np.tan(theta / 2))

#     # ETA_MAX = 4
#     # eta_cut = np.abs(eta) < ETA_MAX
#     # x = x * np.nan_to_num(eta_cut[:,None])


#     energies = x[:,0]
#     total_energy = np.sum(energies)
#     x[:,0] = energies / total_energy
#     x[:,1] = x[:,1] / total_energy
#     x[:,2] = x[:,2] / total_energy
#     x[:,3] = x[:,3] / total_energy

#     # Sort x by x[:,0] in descending order
#     x = x[np.argsort(x[:,0])[::-1]]

    # dataset_open[i] = x

dataset_open = dataset_open[:, :250]
dataset_open = dataset_open[:, :, :4]

# Normalize such that each event has an energy of 1
dataset_open[:, :, 1] /= np.sum(dataset_open[:, :, 0], axis=-1)[:, None]
dataset_open[:, :, 2] /= np.sum(dataset_open[:, :, 0], axis=-1)[:, None]
dataset_open[:, :, 3] /= np.sum(dataset_open[:, :, 0], axis=-1)[:, None]
dataset_open[:, :, 0] /= np.sum(dataset_open[:, :, 0], axis=-1)[:, None]



N = dataset_open.shape[0]
N_particles = dataset_open.shape[1]
print(f"Loaded {N} events with {N_particles} particles each")




# %%%%%%%%%% Load Observables %%%%%%%%%%

jet_observables_dict = build_event_observables()

print(jet_observables_dict.keys())




observable_keys = ["sDipole",] #, ["sThrust", "sDipole"] #["spRinginess", "spIsotropy", "sThrust", "sDipole"]
observable_names = ["dipole",]#, ["thrust", "dipole"] #["ring", "isotropy", "thrust", "dipole"]


# %%%%%%%%%% Compute Observables %%%%%%%%%%



batch = 0
for batch_start in range(0, dataset_open.shape[0], batch_size):


    batch_end = batch_start + batch_size
    dataset = dataset_open[batch_start:batch_end]

    for o, observable_key in enumerate(observable_keys):

        observable = jet_observables_dict[observable_key]
        observable_name = observable_names[o]
        emds, params, loss_history, params_history = observable.compute(dataset, learning_rate= lr, early_stopping=25, N_sample = 75, finite_difference=False, epochs = epochs)

        print(observable_key, emds)

        # save
        np.save(f"{this_dir}/Data/{this_study}_{observable_name}_emds_{batch}.npy", emds)
        np.save(f"{this_dir}/Data/{this_study}_{observable_name}_params_{batch}.npy", params)
        np.save(f"{this_dir}/Data/{this_study}_{observable_name}_loss_history_{batch}.npy", loss_history)
        np.save(f"{this_dir}/Data/{this_study}_{observable_name}_params_history_{batch}.npy", params_history)

        # Try exact computation
        try: 
            hard_emds, hard_params = observable.hard_compute(dataset)
            np.save(f"{this_dir}/Data/{this_study}_{observable_name}_hard_emds_{batch}.npy", hard_emds)
            np.save(f"{this_dir}/Data/{this_study}_{observable_name}_hard_params_{batch}.npy", hard_params)
        except Exception as e:
            print(f"Error in hard_compute: {e}")


    batch += 1
