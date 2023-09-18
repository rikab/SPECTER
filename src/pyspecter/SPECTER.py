# 


# Standard imports
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# JAX imports
import jax
import jax.numpy as jnp
import jax_opt

# Spectral EMD Helper imports
from .SpectralEMD_Helper import weighted_sum, cross_term, theta, ds2



# ################################### #
# ########## SPECTER CLASS ########## #
# ################################### #

class SPECTER():

    def __init__(self) -> None:
        self.obserables = []


    # Function to add an observable to the list of observables
    def add_observable(self, observable):
        """Function to add an observable to the list of observables

        Args:
            observable (Observable): Observable to add to the list of observables
        """
        self.obserables.append(observable)

    # Function to add a list of observables to the list of observables
    def add_observables(self, observables):
        """Function to add a list of observables to the list of observables

        Args:
            observables (list): List of observables to add to the list of observables
        """
        self.obserables.extend(observables)


    # @jads2x.jit
    def compute_spectral_representation(events, omega_max = 2, beta = 1, dtype = jnp.float32):
        """Function to compute the spectral representation of a set of events

        Args:
            events (ndarray): Array of events with shape (batch_size, pad, 3)
            omega_max (float, optional): Maximum omega value. Defaults to 2.
            beta (float, optional): Beta value for the spectral representation. Defaults to 1.
            dtype (jax.numpy.dtype, optional): Data type for the output. Defaults to jax.numpy.float32.

        Returns:
            ndarray: Spectral representation of the events with shape (batch_size, pad*(pad-1)/2, 2)
        """

        # Events shape is (batch_size, pad, 3)
        points, zs = events[:,:,1:], events[:,:,0]
        batch_size = zs.shape[0]
        euclidean_distance_squared = jnp.sum(jnp.square(points[:,:, None, :] - points[:,None, :, :]), axis=-1)
        

        # Upper Triangle Matrices
        omega_ij = jnp.triu(euclidean_distance_squared, k = 1)
        triangle_indices = jnp.triu_indices(zs.shape[1], k = 1)
        triangle_indices_i = triangle_indices[0]
        triangle_indices_j = triangle_indices[1]

        # Get pairwise products of energies
        ee_ij = jnp.triu(zs[:,:,None] * zs[:,None,:])
        ee2 = jnp.trace(ee_ij, axis1 = 1, axis2=2)

        # Flatten to 1D Spectral Representation and remove 0s
        omega_n = omega_ij[:,triangle_indices_i, triangle_indices_j]
        omega_n = jnp.power(omega_n, beta) / beta
        ee_n = 2 * ee_ij[:,triangle_indices_i, triangle_indices_j]


        s = jnp.stack((omega_n, ee_n), axis = 1)
        s = jnp.transpose(s, (0,2,1))
        
        # Sort and append 0
        indices = s[:,:,0].argsort()
        temp_indices = jnp.arange(batch_size)[:,jnp.newaxis]
        s = s[temp_indices,indices]
        s0 = jnp.zeros((batch_size, 1, 1))
        s1 = jnp.concatenate((s0, jnp.expand_dims(ee2, axis = (1,2))), axis = 2)
        s = jnp.concatenate((s1, s), axis = 1)

        return s.astype(dtype)
    

    def spectralEMD(self, events1, events2, type1 = "events", type2 = "events", beta = 1):
        """Function to compute the spectral Earth Mover's Distance between two sets of events

        Args:
            events1 (ndarray): Array of events with shape (batch_size, pad1, 3), or spectral representation with shape (batch_size, pad1*(pad1-1)/2, 2)
            events2 (ndarray): Array of events with shape (batch_size, pad2, 3), or spectral representation with shape (batch_size, pad2*(pad2-1)/2, 2)
            type1 (str, optional): Type of events1. Must be either "spectral" or "events". Defaults to "events".
            type2 (str, optional): Type of events2. Must be either "spectral" or "events". Defaults to "events".
            beta (float, optional): Beta value for the spectral representation. Defaults to 1. Unused if type1 and type2 are "spectral".

        Returns:
            ndarray: Spectral EMD between events1 and events2 of size (batch_size,)
        """

        # ########## Input formatting and validation ##########

        # Compute spectral representation
        if type1 == "events":
            s1 = self.compute_spectral_representation(events1, beta = beta)
        elif type1 == "spectral":
            s1 = events1
        else:
            raise ValueError(f"Invalid type {type1} for events1! Must be 'events' or 'spectral'!")
        
        if type2 == "events":
            s2 = self.compute_spectral_representation(events2, beta = beta)
        elif type2 == "spectral":
            s2 = events2
        else:
            raise ValueError(f"Invalid type {type2} for events2! Must be 'events' or 'spectral'!")

        # Check batch sizes
        batch_size_1 = s1.shape[0]
        batch_size_2 = s2.shape[0]
        if batch_size_2 != batch_size_1:
            raise ValueError("Must have equal batch sizes for line-by-line sEMDs! Found batch sizes of {batch_size_1} and {batch_size_2}!")


        # Compute EMD
        return self.EMD(s1, s2, type1, type2)
    

    # Compilation handling
    def compile(self, verbose = True):
        """Function to compile the SPECTER model. This converts expressions to efficient XLA for evalulation. 
           Each function is then run once on test events to ensure that the jaxpr expressions are fully traced before compilation.
        """

        # ########## Compile individual functions ##########

        if verbose:
            print("Compiling SPECTER model...")

        self.compute_spectral_representation = jax.jit(self.compute_spectral_representation)
        self.ds2 = jax.jit(ds2)
        self.ds2_events1_events2, self.ds2_events1_s2 = self.initialize_ds2_events_functions()
        self.ds2_events1_events2 = jax.jit(self.ds2_events1_events2)
        self.ds2_events1_s2 = jax.jit(self.ds2_events1_s2)

        self.compiled_functions = [self.compute_spectral_representation, self.ds2, self.ds2_events1_events2, self.ds2_events1_s2]

        
        # ########## JAX Tracing ##########

        # Test events
        test_events_1 = jnp.ones((3, 99, 3)) 
        test_events_2 = jnp.ones((3, 101, 3))

        # Run each function once to ensure that the jaxpr expressions are fully traced before compilation
        test_s_1 = self.compute_spectral_representation(test_events_1).block_until_ready()
        test_s_2 = self.compute_spectral_representation(test_events_2).block_until_ready()

        test_ds2 = self.ds2(test_s_1, test_s_2).block_until_ready()
        test_ds2_events1_events2 = self.ds2_events1_events2(test_events_1, test_events_2).block_until_ready()
        test_ds2_events1_s2 = self.ds2_events1_s2(test_events_1, test_s_2).block_until_ready()

        # Initialize gradient functions for each compiled function
        self.spectral_representation_gradients = self.initialize_gradient_function(self.compute_spectral_representation)
        self.ds2_gradients = self.initialize_gradient_function(self.ds2)
        self.ds2_events1_events2_gradients = self.initialize_gradient_function(self.ds2_events1_events2)
        self.ds2_events1_s2_gradients = self.initialize_gradient_function(self.ds2_events1_s2)

        # Trace through each gradient function
        test_spectral_representation_gradients = self.spectral_representation_gradients(test_events_1).block_until_ready()
        test_ds2_gradients = self.ds2_gradients(test_s_1, test_s_2).block_until_ready()
        test_ds2_events1_events2_gradients = self.ds2_events1_events2_gradients(test_events_1, test_events_2).block_until_ready()
        test_ds2_events1_s2_gradients = self.ds2_events1_s2_gradients(test_events_1, test_s_2).block_until_ready()


    


    # ###################################### #
    # ########## HELPER FUNCTIONS ########## #
    # ###################################### #


    # Function to initialize the gradient function for a given function
    def initialize_gradient_function(function):
        """Helper function to initialize the gradient function for a given function

        Args:
            function (function): Function to initialize the gradient function for

        Returns:
            function: Batchwise Gradient function for the given function
        """

        return jax.jit(jax.grad(jnp.sum(function), argnums=0))
    
    # Function to initialize the ds2_events1_events2 and ds2_events1_s2 functions
    def initialize_ds2_events_functions(self):
        """Helper function to initialize the ds2_events1_events2 and ds2_events1_s2 functions

        Returns:
            function, function: ds2_events1_events2 and ds2_events1_s2 functions
        """



        def ds2_events1_events2(events1, events2):

            s1 = self.compute_spectral_representation(events1)
            s2 = self.compute_spectral_representation(events2)

            return self.ds2(s1, s2)
        
        def ds2_events1_s2(events1, s2):

            s1 = self.compute_spectral_representation(events1)

            return self.ds2(s1, s2)
    
        return ds2_events1_events2, ds2_events1_s2