# Standard Imports
import numpy as np
from time import time
from matplotlib import pyplot as plt
import os


# Utils
from pyspecter.utils.plot_utils import newplot, plot_event


# SHAPER
from pyshaper.CommonObservables import buildCommmonObservables
from pyshaper.Observables import Observable
from pyshaper.Shaper import Shaper

# Necessary GPU nonsense
import torch 

if torch.cuda.is_available():  
    dev = "cuda:0" 
    print("Using GPU!")
else:  
    dev = "cpu"  
    print("Using CPU!")
device = torch.device(dev) 

# Parameters 
N = 100000
batch_size = 10000
R = 1
this_dir = ""
this_study = "cmsopendata"

data_directory = "/n/holyscratch01/iaifi_lab/rikab/top_qcd/top_qcd_0.npz"
this_dir = "/n/home01/rikab/SPECTER_STUDIES"

d = np.load(data_directory)
X = np.array(d["data"])
Y = d["labels"]

qcd = X[Y == 0]
top = X[Y == 1]

dataset = qcd[:N, :75, :3]

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
        phis = x[:,2]
        phis = (phis ) % (2 * np.pi)
        phis_above_pi = phis > np.pi
        phis[phis_above_pi] = phis[phis_above_pi] - 2 * np.pi
        phis_below_minus_pi = phis < - np.pi
        phis[phis_below_minus_pi] = phis[phis_below_minus_pi] + 2 * np.pi
        x[:,2] = phis - np.average(phis, weights=energies)

print(dataset.shape)


# Common Observables
commonObservables, pointers = buildCommmonObservables(N = 3, beta = 2, R = R, device = device)
_1ringiness = commonObservables["1-Ringiness"]
_1point_ringiness = commonObservables["1-Point-Ringiness"]
_1diskiness = commonObservables["1-Diskiness"]
_1point_diskiness = commonObservables["1-Point-Diskiness"]
_1lineliness = commonObservables["1-Ellipsiness"]
_1lineliness.freeze("Radius2", torch.tensor([0.0001,]))

_1pronginess = commonObservables["1-Subjettiness"]
_2pronginess = commonObservables["2-Subjettiness"]
_3pronginess = commonObservables["3-Subjettiness"]



# Set defaults
_1ringiness.params["Radius"].default_value = torch.tensor((R , ))
_1point_ringiness.params["Radius"].default_value = torch.tensor((R , ))
_1diskiness.params["Radius"].default_value = torch.tensor((R , ))
_1point_diskiness.params["Radius"].default_value = torch.tensor((R , ))
_1lineliness.params["Radius1"].default_value = torch.tensor((R , ))


observables = {}
observables["1-Ringiness"] = _1ringiness
# observables["1-Point-Ringiness"] = _1point_ringiness
observables["1-Diskiness"] = _1diskiness
# observables["1-Point-Diskiness"] = _1point_diskiness
observables["1-Ellipsiness"] = _1lineliness
observables["1-Subjettiness"] = _1pronginess
observables["2-Subjettiness"] = _2pronginess
observables["3-Subjettiness"] = _3pronginess


# RUN 

# Initialize SHAPER
shaper = Shaper(observables, device)
shaper.to(device)

for (i, batch) in enumerate(range(0, dataset.shape[0], batch_size)):

    batch_start = batch
    batch_end = batch_start + batch_size
    dataset_batch = dataset[batch_start:batch_end]

    dataset_emds, dataset_params = shaper.calculate(dataset_batch, epochs = 500, verbose=True, lr = 0.0025, N = 150, scaling = 0.95, epsilon = 0.001, early_stopping= 50)

    # Save the results
    folder = os.path.join(this_dir, "Data/SHAPER")
    for observable in observables.keys():

        EMDs = []
        params = []

        for j in range(dataset_batch.shape[0]):
            e = dataset_params[observable][j]["EMD"]
            p = dataset_params[observable][j]

        np.save(os.path.join(folder, observable + f"_emds_{i}"), dataset_emds[observable])
        np.save(os.path.join(folder, observable + f"_params_{i}"), dataset_params[observable])


# Combine all batched numpy files
for observable in observables.keys():

    emds = []
    params = []

    for i in range(10):
        emds.append(np.load(os.path.join(folder, observable + f"_emds_{i}.npy"), allow_pickle=True))
        params.append(np.load(os.path.join(folder, observable + f"_params_{i}.npy"), allow_pickle=True))

    emds = np.concatenate(emds, axis = 0)
    params = np.concatenate(params, axis = 0)

    np.save(os.path.join(folder, observable + "_emds"), emds)
    np.save(os.path.join(folder, observable + "_params"), params)

# dataset_emds, dataset_params = shaper.calculate(dataset, epochs = 500, verbose=True, lr = 0.005, N = 150, scaling = 0.95, epsilon = 0.001, early_stopping= 25)

# # Save the results
# folder = os.path.join(this_dir, "Data/SHAPER")
# for observable in observables.keys():

#     EMDs = []
#     params = []

#     for j in range(dataset.shape[0]):
#         e = dataset_params[observable][j]["EMD"]
#         p = dataset_params[observable][j]

#     np.save(os.path.join(folder, observable + "_emds"), dataset_emds[observable])
#     np.save(os.path.join(folder, observable + "_params"), dataset_params[observable])

