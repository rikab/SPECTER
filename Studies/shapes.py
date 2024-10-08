# Standard Imports
import numpy as np
from time import time
from matplotlib import pyplot as plt
import os


from pyspecter.SPECTER import SPECTER
from pyspecter.CommonObservables import build_jet_observables
# from pyspecter.SpecialObservables import SpecialObservables

# Utils
try:
    from rikabplotlib.plot_utils import newplot, plot_event
except:
    from pyspecter.utils.plot_utils import newplot, plot_event




# %%%%%%%%%% Load Data %%%%%%%%%%

# Parameters 
R = 1.0
this_dir = ""
this_study = "cmsopendata"
lr = 0.005
epochs = 1500

batch_size = 10000


data_directory = "/n/holyscratch01/iaifi_lab/rikab/top_qcd/top_qcd_0.npz"
this_dir = "/n/home01/rikab/SPECTER_STUDIES"

d = np.load(data_directory)
X = np.array(d["data"])
Y = d["labels"]

qcd = X[Y == 0]
top = X[Y == 1]

dataset = qcd[:,:125,:3]
print(dataset.shape)


# %%%%%%%%%% Load Observables %%%%%%%%%%

jet_observables_dict = build_jet_observables(R = R)

print(jet_observables_dict.keys())

# _splineliness = jet_observables_dict["spLineliness"]
# _springiness = jet_observables_dict["spRinginess"]
# _sdiskiness = jet_observables_dict["spDiskiness"]
_1spronginess = jet_observables_dict["1-sPronginess"]
# _2spronginess = jet_observables_dict["2-sPronginess"]
# _3spronginess = jet_observables_dict["3-sPronginess"]



observable_keys = ["spLineliness", "spRinginess", "spDiskiness", "1-sPronginess", "2-sPronginess", "3-sPronginess"]
observable_names = ["line","ring", "disk", "1sprong", "2sprong", "3sprong"]


# %%%%%%%%%% Compute Observables %%%%%%%%%%



batch = 0
for batch_start in range(0, dataset.shape[0], batch_size):


    batch_end = batch_start + batch_size
    dataset = qcd[batch_start:batch_end, :75, :3]

    for x in dataset:

        energies = x[:,0]
        etas = x[:,1]
        phis = x[:, 2]

        # translate such that the phi of the hardest particle is 0
        max_phi_index = np.argmax(energies)
        phis = phis - phis[max_phi_index]


        # # # Wrap the phis to be between -pi and pi
        phis = (phis ) % (2 * np.pi)
        phis_above_pi = phis > np.pi
        phis[phis_above_pi] = phis[phis_above_pi] - 2 * np.pi
        phis_below_minus_pi = phis < - np.pi
        phis[phis_below_minus_pi] = phis[phis_below_minus_pi] + 2 * np.pi
        # # phis = (phis ) % (2 * np.pi)

        x[:,0] = energies / np.sum(energies)
        x[:,1] = etas - np.average(etas, weights=energies)
        x[:,2] = phis - np.average(phis, weights=energies)


        # # # Wrap the phis to be between -pi and pi
        for i in range(10):

            phis = x[:,2]
            phis = (phis ) % (2 * np.pi)
            phis_above_pi = phis > np.pi
            phis[phis_above_pi] = phis[phis_above_pi] - 2 * np.pi
            phis_below_minus_pi = phis < - np.pi
            phis[phis_below_minus_pi] = phis[phis_below_minus_pi] + 2 * np.pi
            x[:,2] = phis - np.average(phis, weights=energies)


    for o, observable_key in enumerate(observable_keys):

        observable = jet_observables_dict[observable_key]
        observable_name = observable_names[o]
        emds, params, loss_history, params_history = observable.compute(dataset, learning_rate= lr, early_stopping=150, N_sample = 125, finite_difference=False, epochs = epochs)

        # save
        np.save(f"{this_dir}/Data/{observable_name}_emds_{batch}.npy", emds)
        np.save(f"{this_dir}/Data/{observable_name}_params_{batch}.npy", params)
        # np.save(f"{this_dir}/Data/{observable_name}_loss_history_{batch}.npy", loss_history)
        # np.save(f"{this_dir}/Data/{observable_name}_params_history_{batch}.npy", params_history)

        # Try exact computation
        try: 
            hard_emds, hard_params = observable.hard_compute(dataset)
            np.save(f"{this_dir}/Data/{observable_name}_hard_emds_{batch}.npy", hard_emds)
            np.save(f"{this_dir}/Data/{observable_name}_hard_params_{batch}.npy", hard_params)
        except Exception as e:
            print(f"Error in hard_compute: {e}")


    batch += 1


# Load data batches and consolodate

# Combine all batched numpy files
for (o, observable) in enumerate(observable_keys):

    observable_name = observable_names[o]

    emds = []
    params = []

    hard_emds = []
    hard_params = []

    for i in range(10):

        emds.append(np.load(f"{this_dir}/Data/{observable_name}_emds_{i}.npy", allow_pickle = True))

        if param_names[o] != "":
            params.append(np.load(f"{this_dir}/Data/{observable_name}_params_{i}.npy", allow_pickle = True).item()[param_names[o]])

        try:
            hard_emds.append(np.load(f"{this_dir}/Data/{observable_name}_hard_emds_{i}.npy", allow_pickle = True))

            if param_names[o] != "":
                hard_params.append(np.load(f"{this_dir}/Data/{observable_name}_hard_params_{i}.npy", allow_pickle = True).item()[param_names[o]])

        except:
            pass

    print(params)
    emds = np.concatenate(emds, axis = 0)

    if param_names[o] != "":
        params = np.concatenate(params, axis = 0) 


    np.save(f"{this_dir}/Data/{observable_name}_emds.npy", emds)
    np.save(f"{this_dir}/Data/{observable_name}_params.npy", params)

    try:


        hard_emds = np.concatenate(hard_emds, axis = 0)

        if param_names[o] != "":
            hard_params = np.concatenate(hard_params, axis = 0) 

        np.save(f"{this_dir}/Data/{observable_name}_hard_emds.npy", hard_emds)
        np.save(f"{this_dir}/Data/{observable_name}_hard_params.npy", hard_params)

    except:
        pass


