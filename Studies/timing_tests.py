
# #############################
# ########## IMPORTS ##########
# #############################


from pyspecter.SPECTER import SPECTER
from pyshaper.Shaper import Shaper
from ot.lp import emd, emd2

# Utils
from pyspecter.utils.data_utils import load_cmsopendata, load_triangles, random_triangles, kT_N


# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Necessary GPU nonsense for SHAPER
import torch 
if torch.cuda.is_available():  
    dev = "cuda:0" 
    print("Using GPU!")
else:  
    dev = "cpu"  
    print("Using CPU!")
device = torch.device(dev) 

import jax
print(f'Jax backend: {jax.default_backend()}')


# ########## PARAMETERS ##########

N_trials = 10
R = 1
beta = 2
run_name = "gpu_base2"

working_dir = "/n/home01/rikab/SPECTER_STUDIES"

batch_sizes = [1,10, 100, 1000, 10000]
particle_sizes = np.logspace(0, 16, 49, base = 2).astype(int)



def generate_unit_disk(N_batch, N_particles, R=1):

    r = np.random.rand(N_batch, N_particles)
    theta = np.random.rand(N_batch, N_particles) * 2 * np.pi
    x = np.sqrt(r) * np.cos(theta) * R / 3 - 3*R/8
    y = np.sqrt(r) * np.sin(theta) * R / 3 + R/8

    weights = np.exp(-3 * np.random.rand(N_batch, N_particles))
    weights /= np.sum(weights, axis=1)[:, None]

    events = np.stack([weights, x, y], axis=-1)

    return events

def generate_gaussian(N_batch, N_particles, R=1):
 
    x = np.random.randn(N_batch, N_particles+ 1) * R/6 + np.random.choice([-R/2, R/2], size=(N_batch, N_particles+ 1))
    y = np.random.randn(N_batch, N_particles+ 1) * R/6

    weights = np.exp(-2 * np.random.rand(N_batch, N_particles+ 1))
    weights /= np.sum(weights, axis=1)[:, None]

    events = np.stack([weights, x, y], axis=-1)

    return events


# ############################
# ########## SPECTER #########
# ############################

# Initialize SPECTER
specter = SPECTER(compile = True)



def test_SPECTER_times(N_trials, N_batch, N_particles, R, beta, run_name):

    times = []



    for i in range(N_trials):
        dataset_1 = generate_unit_disk(N_batch, N_particles, R)
        dataset_2 = generate_gaussian(N_batch, N_particles, R)
        start = time.time()
        specter_emds = specter.spectralEMD(dataset_1, dataset_2)
        end = time.time()
        times.append(end - start)

    times = np.array(times)
    print(f"Average time for {N_trials} trials with {N_batch} batches and {N_particles} particles: {np.mean(times)}")

    filename = os.path.join(working_dir, f"Data/SPECTER_{N_trials}_{N_batch}_{N_particles}_{run_name}_times.npy")
    np.save(filename, times)


    return times



# ############################
# ########## SHAPER ##########
# ############################

shaper = Shaper({}, device)
shaper.to(device)


def test_SHAPER_times(N_trials, N_batch, N_particles, R, beta, run_name):

    times = []

    for i in range(N_trials):
        dataset_1 = generate_unit_disk(N_batch, N_particles, R)
        dataset_2 = generate_gaussian(N_batch, N_particles, R)
        start = time.time()
        shaper_emds_no_isometry = shaper.pairwise_emds2(dataset_1, dataset_2, R = R, beta = 2, epsilon = 0.001, scaling = 0.95)
        end = time.time()
        times.append(end - start)

    times = np.array(times)
    print(f"Average time for {N_trials} trials with {N_batch} batches and {N_particles} particles: {np.mean(times)}")

    filename = os.path.join(working_dir, f"Data/SHAPER_{N_trials}_{N_batch}_{N_particles}_{run_name}_times.npy")
    np.save(filename, times)

    return times


# ########## PYTHON OPTIMAL TRANSPORT ##########

print(emd2)

def _cdist_phi_y(X,Y, ym):
        # define ym as the maximum rapidity cut on the quasi-isotropic event
        # Make sure the phi values are in range                                                                                                                                          
        phi1 = (X[:,2])
        phi2 = (Y[:,2])

        y1 = X[:,1]
        y2 = Y[:,1]

        # Return an array of particle distances 
        phi_d = np.abs(phi1[:,np.newaxis] - phi2)
        y_d = np.abs(y1[:,np.newaxis] - y2)

        # Calculate the distance in the y-phi plane
        dist = np.power(phi_d**2 + y_d**2, beta / 2)


        return dist


def test_POT_times(N_trials, N_batch, N_particles, R, beta, run_name):
        
        times = []

        for i in range(N_trials):
            dataset_1 = generate_unit_disk(1, N_particles, R)
            dataset_2 = generate_gaussian(1, N_particles, R)
            start = time.time()
            for (event1, event2) in zip(dataset_1, dataset_2):
                ai = event1[:,0]
                bj = event2[:,0]
                M = _cdist_phi_y(event1, event2, R)
                ai = ai.astype(np.float64) 
                ai = (ai / ai.sum().astype(np.float64)).astype(np.float64)
                emd_val, log = emd2(ai, bj.astype(np.float64)/ bj.sum().astype(np.float64), M,log=True,numItermax = 10000000)
                # Should only return 0 when two events are identical. If returning 0 otherwise, problems in config
            end = time.time()
            times.append(N_batch * (end - start))

     
        times = np.array(times)
        print(f"Average time for {N_trials} trials with {N_batch} batches and {N_particles} particles: {np.mean(times)}")

        filename = os.path.join(working_dir, f"Data/POT_{N_trials}_{N_batch}_{N_particles}_{run_name}_times.npy")
        np.save(filename, times)









# Run SPECTER
for N_particles in particle_sizes:
    for N_batch in batch_sizes:

        try:
            test_SPECTER_times(N_trials, N_batch, N_particles, R, beta, "cpu")
        except Exception as e:
            print(f"Failed for {N_batch} batches and {N_particles} particles")
            print(e)

            
print("SPECTER done --------------")

# Run SHAPER
for N_particles in particle_sizes:
    for N_batch in batch_sizes:

        try:
            test_SHAPER_times(N_trials, N_batch, N_particles, R, beta, "cpu")
        except Exception as e:
            print(f"Failed for {N_batch} batches and {N_particles} particles")
            print(e)

print("SHAPER done --------------")



# Run POT
for N_particles in particle_sizes:
    for N_batch in batch_sizes:

        try:
            test_POT_times(N_trials, N_batch, N_particles, R, beta, "cpu")
        except Exception as e:
            print(f"Failed for {N_batch} batches and {N_particles} particles")
            print(e)


print("POT done --------------")