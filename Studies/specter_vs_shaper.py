#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspecter.SPECTER import SPECTER
from pyshaper.Shaper import Shaper

# Utils
from pyspecter.utils.data_utils import  kT_N
from rikabplotlib.plot_utils import newplot, plot_event


# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# Necessary GPU nonsense for SHAPER
import torch 
if torch.cuda.is_available():  
    dev = "cuda:0" 
    print("Using GPU!")
else:  
    dev = "cpu"  
    print("Using CPU!")
device = torch.device(dev) 


# In[2]:


def minkowski_dot(a, b):
    return a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)

def generate_RAMBO(num_events, num_particles, total_energy, seed=None, return_weight = True):

    # Set seed if given
    if seed is not None:
        np.random.seed(seed)


    # Initial c, phi, e values from uniform distributions
    cos_theta = np.random.uniform(-1, 1, (num_events, num_particles))
    phi = np.random.uniform(0, 2 * np.pi, (num_events, num_particles))
    energy = -np.log(np.random.uniform(0, 1, (num_events, num_particles, 2)).prod(axis=2))
    mom_x = energy * np.sqrt(1 - cos_theta**2) * np.cos(phi)
    mom_y = energy * np.sqrt(1 - cos_theta**2) * np.sin(phi)
    mom_z = energy * cos_theta
    initial_momenta = np.stack([energy, mom_x, mom_y, mom_z], axis=-1)

    # Total momentum and invariant mass for each event
    total_momentum = initial_momenta.sum(axis=1)
    invariant_mass = np.sqrt(minkowski_dot(total_momentum, total_momentum))
    boost_vector = -total_momentum / invariant_mass[:, np.newaxis]
    scale_factor = total_energy / invariant_mass
    gamma_factor = total_momentum[:, 0] / invariant_mass
    a_factor = 1 / (1 + gamma_factor)

    # Rescaling and boosting for each event (vectorized)
    final_momenta = []
    for i in range(num_events):
        event_momenta = []
        for momentum in initial_momenta[i]:
            dot_boost_mom = np.dot(boost_vector[i:i+1, 1:], momentum[1:])
            energy = scale_factor[i] * (gamma_factor[i] * momentum[0] + dot_boost_mom)
            mom_x = scale_factor[i] * (momentum[1] + boost_vector[i, 1] * (momentum[0] + a_factor[i] * dot_boost_mom))
            mom_y = scale_factor[i] * (momentum[2] + boost_vector[i, 2] * (momentum[0] + a_factor[i] * dot_boost_mom))
            mom_z = scale_factor[i] * (momentum[3] + boost_vector[i, 3] * (momentum[0] + a_factor[i] * dot_boost_mom))
            event_momenta.append([energy, mom_x, mom_y, mom_z])
        final_momenta.append(event_momenta)

    # Convert to numpy array
    final_momenta = np.array(final_momenta)[:,:,:,0]

    if not return_weight:
        return final_momenta
    
    else:

        # Calculate weights for phase space volume
        weights = ((total_energy ** 2) ** (num_particles - 2) * np.ones_like(scale_factor)) /  ((2 * np.pi) ** (3 * num_particles - 4)) * (np.pi / 2) ** (num_particles - 1) / (scipy.special.gamma(num_particles) * scipy.special.gamma(num_particles - 1))

        return final_momenta, weights / num_events


def boost(four_momenta, boost_vector):
    """
    Boosts a set of four momenta by a given boost vector
    """
    # Calculate gamma factor
    gamma = 1 / np.sqrt(1 - np.sum(boost_vector ** 2))

    # Calculate dot products
    dot_products = np.sum(four_momenta[:, 1:] * boost_vector, axis=1)

    # Calculate new energies and momenta
    energy = gamma * (four_momenta[:, 0] + dot_products)
    momenta = four_momenta[:, 1:] + (gamma - 1) * dot_products[:, np.newaxis] * boost_vector + gamma * energy[:, np.newaxis] * boost_vector

    return np.stack([energy, *momenta.T], axis=-1)


def generate_RAMBO_jets(n_events, n_particles):

    sqrt_s = 150
    jets = generate_RAMBO(n_events, n_particles, sqrt_s, return_weight = False, seed = 42)

    # Boost along the x direction such that the jet has an energy of 500 GeV
    jet_energy = 500
    desired_center_of_mass_momentum = np.array([jet_energy, np.sqrt(jet_energy**2 - sqrt_s**2), 0, 0])
    boost_vector = desired_center_of_mass_momentum[1:] / desired_center_of_mass_momentum[0]

    data = np.zeros((n_events, n_particles, 3))

    for (i,jet) in enumerate(jets):
        jet = boost(jet, boost_vector)

        pt = np.sqrt(jet[:, 1] ** 2 + jet[:, 2] ** 2)
        theta = np.arccos(jet[:, 3] / np.sqrt(pt**2 + jet[:, 3]**2))
        eta = 0.5 * np.log((jet[:, 0] + jet[:, 3]) / (jet[:, 0] - jet[:, 3]))
        phi = np.arctan2(jet[:, 2], jet[:, 1])

        data[i] = np.stack([pt, phi, theta], axis=-1)

        # normalize pt and center eta, phi
        data[i, :, 0] /= data[i, :, 0].sum()
        data[i, :, 1] -= np.average(data[i, :, 1], weights=data[i, :, 0])
        data[i, :, 2] -= np.average(data[i, :, 2], weights=data[i, :, 0])

    return data






# In[4]:


# CMS Open Sim Parameters
R = 0.5
pt_lower = 475
pt_upper = 525
eta_cut = 1.9
quality = 2
pad = 100 # Note that runtime is pad^2, memory is pad^4


R = 0.5
pt_lower = 475
pt_upper = 525
eta_cut = 1.9
quality = 2
pad = 100 # Note that runtime is pad^2, memory is pad^4


N_jets = 25
dataset_name = "qcd"
n_samples = 15000*2
batch_size = 1 # Number of pairs to process in parallel, can probably be increased.


def center_and_normalize(dataset):

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


data_directory = "/n/holyscratch01/iaifi_lab/rikab/top_qcd/top_qcd_0.npz"
this_dir = "/n/home01/rikab/SPECTER_STUDIES"

d = np.load(data_directory)
X = np.array(d["data"])
Y = d["labels"]

qcd = X[Y == 0][:n_samples * 2, :75, :3]
top = X[Y == 1][:n_samples * 2, :75, :3]
rambo = generate_RAMBO_jets(n_samples * 2, 75)
np.save(f"Data/rambo.npy", rambo)


center_and_normalize(qcd)
center_and_normalize(top)


if dataset_name == "qcd":

    dataset = qcd
    weights = np.ones_like(dataset)


elif dataset_name == "top":

    dataset = top
    weights = np.ones_like(dataset)


elif dataset_name == "RAMBO":

    rambo_jets = generate_RAMBO_jets(n_samples, N_jets)

    # save rambo jets
    np.save(f"{this_dir}/Data/rambo_{N_jets}jets.npy",rambo_jets)

    dataset = rambo_jets
    weights = np.ones_like(dataset)





# In[ ]:




# Set up and compile SPECTER
specter = SPECTER(compile = True)


def rotate(dataset, angle):

    energies, xs, ys = dataset[:,:,0], dataset[:,:,1], dataset[:,:,2]
    new_xs = xs*np.cos(angle) - ys*np.sin(angle)
    new_ys = xs*np.sin(angle) + ys*np.cos(angle)

    return np.stack([energies, new_xs, new_ys], axis = 2)




def run_jets(N_jets, dataset, dataset_name):


    dataset_jets = kT_N(dataset, N_jets, R)

    # Recenter 
    for x in dataset_jets:



        pt_weights = x[:,0]
        x[:,1] = x[:,1] - np.average(x[:,1], weights=pt_weights)
        x[:,2] = x[:,2] - np.average(x[:,2], weights=pt_weights)

    
    # Split dataset into 2 halves
    dataset1 = dataset_jets[:n_samples//2]
    dataset2 = dataset_jets[n_samples//2:n_samples]
    weights1 = weights[:n_samples//2]
    weights2 = weights[n_samples//2:n_samples]



    start = time.time()
    specter_emds = specter.spectralEMD(dataset1, dataset2)
    end = time.time()
    print("SPECTER took {} seconds".format(end - start))

    # Set up and compile SHAPER
    shaper = Shaper({}, device)
    shaper.to(device)




    start = time.time()

    num_angles = 180
    angles = np.linspace(0, 2*np.pi, num_angles)
    emds = np.zeros((dataset1.shape[0],num_angles))

    for i, angle in tqdm(enumerate(angles)):
        dataset2_rotated = rotate(dataset2, angle)
        emds[:,i] = shaper.pairwise_emds2(dataset1, dataset2_rotated, R = R, beta = 2, epsilon = 0.001, scaling = 0.95)
        
        min_emd_so_far = emds[:,:(i+1)].min(axis = 1)

    shaper_emds = emds.min(axis = 1)
    print(f"SHAPER took {time.time() - start} seconds")
    end = time.time()



    np.save(f"Data/{dataset_name}_specter_emds_{N_jets}jets.npy", specter_emds)
    np.save(f"Data/{dataset_name}_shaper_emds_{N_jets}jets.npy", shaper_emds)


# In[ ]:


num_jets = [1, 2, 3, 5, 10, 25]
for N_jets in num_jets:


    run_jets(N_jets, rambo, "RAMBO")
    run_jets(N_jets, qcd, "qcd")
    run_jets(N_jets, top, "top")
