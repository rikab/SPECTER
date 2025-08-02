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

# Observables
from pyspecter.Observables import Observable

# Spectral EMD Helper imports
from pyspecter.SpectralEMD_Helper import compute_spectral_representation, compute_spectral_representation_cylindrical,  ds2
from pyspecter.SpectralEMD_Helper import cross_ds2_events1_events2
from pyspecter.SpectralEMD_Helper import balance



# ################################### #
# ########## SPECTER CLASS ########## #
# ################################### #

class SPECTER():

    def __init__(self, observables = None, compile = True) -> None:
        """Class to compute the Spectral Earth Mover's Distance between two sets of events.

        Args:
            observables (list, optional): List of observables to use in the SPECTER model. Defaults to None.
            compile (bool, optional): Whether or not to compile the SPECTER model. Defaults to True.
        """

        
        # Compile the model
        if compile:
            self.compile()
            self.compiled = True
        else:
            self.compiled = False

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
    def compute_spectral_representation(self, events, omega_max = 2, beta = 1, dtype = jnp.float32):
        """Function to compute the spectral representation of a set of events. Must be compiled before use on batched events -- see SPECTER.compile().

        Args:
            events (ndarray): Array of events with shape (batch_size, pad, 3)
            omega_max (float, optional): Maximum omega value. Defaults to 2.
            beta (float, optional): Beta value for the spectral representation. Defaults to 1.
            dtype (jax.numpy.dtype, optional): Data type for the output. Defaults to jax.numpy.float32.

        Returns:
            ndarray: Spectral representation of the events with shape (batch_size, pad*(pad-1)/2, 2)
        """

        return compute_spectral_representation(events, omega_max, beta, dtype)
    
    def compute_spectral_representation_spherical(self, events, omega_max = 2, beta = 1, dtype = jnp.float32):
        """Function to compute the spectral representation of a set of events. Must be compiled before use on batched events -- see SPECTER.compile().

        Args:
            events (ndarray): Array of events with shape (batch_size, pad, 3)
            omega_max (float, optional): Maximum omega value. Defaults to 2.
            beta (float, optional): Beta value for the spectral representation. Defaults to 1.
            dtype (jax.numpy.dtype, optional): Data type for the output. Defaults to jax.numpy.float32.

        Returns:
            ndarray: Spectral representation of the events with shape (batch_size, pad*(pad-1)/2, 2)
        """

        return compute_spectral_representation(events, omega_max, beta, dtype, euclidean=False)
    

    def compute_spectral_repsentation_cylindrical(self, events, omega_max = 2, beta = 1, dtype = jnp.float32):
        """Function to compute the spectral representation of a set of events. Must be compiled before use on batched events -- see SPECTER.compile().

        Args:
            events (ndarray): Array of events with shape (batch_size, pad, 3)
            omega_max (float, optional): Maximum omega value. Defaults to 2.
            beta (float, optional): Beta value for the spectral representation. Defaults to 1.
            dtype (jax.numpy.dtype, optional): Data type for the output. Defaults to jax.numpy.float32.

        Returns:
            ndarray: Spectral representation of the events with shape (batch_size, pad*(pad-1)/2, 2)
        """

        return compute_spectral_representation_cylindrical(events, omega_max, beta, dtype)
    
    # Function to compute the spectral representation of a set of events. Must be compiled before use on batched events -- see SPECTER.compile().
    def spectralEMD(self, events1, events2, type1 = "events", type2 = "events", beta = 1, metric = "euclidean", omega_max = None, verbose = False):
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

        start_time = time.time()

        # ########## Input formatting and validation ##########

        # Format and validate inputs
        if type1 == "events":
            if metric == "euclidean":
                s1 = self.compute_spectral_representation(events1)
            elif metric == "spherical":
                s1 = self.compute_spectral_representation_spherical(events1)
            elif metric == "cylindrical":
                s1 = self.compute_spectral_repsentation_cylindrical(events1)
        elif type1 == "spectral":
            s1 = events1
        else:
            raise ValueError(f"Invalid type {type1} for events1! Must be 'events' or 'spectral'!")
        
        if type2 == "events":
            if metric == "euclidean":
                s2 = self.compute_spectral_representation(events2)
            elif metric == "spherical":
                s2 = self.compute_spectral_representation_spherical(events2)
            elif metric == "cylindrical":
                s2 = self.compute_spectral_repsentation_cylindrical(events2)
        elif type2 == "spectral":
            s2 = events2
        else:
            raise ValueError(f"Invalid type {type2} for events2! Must be 'events' or 'spectral'!")
        

        # Check for balanced OT
        total_energy_1 = jnp.sum(events1[:, :, 0], axis = 1)
        total_energy_2 = jnp.sum(events2[:, :, 0], axis = 1)


        tolerance = 1e-4
        if jnp.any(jnp.abs(total_energy_1 - total_energy_2) > tolerance):

            # get the indices of the events that are not balanced
            indices = jnp.where(jnp.abs(total_energy_1 - total_energy_2) > tolerance)[0]
            # get the values of the events that are not balanced

            if omega_max is None:

                indices = jnp.where(jnp.abs(total_energy_1 - total_energy_2) > tolerance)[0]
                raise ValueError(f"Events are not balanced to within 1e-4! Indices: {indices}. Total energy of events1: {total_energy_1[indices]} and events2: {total_energy_2[indices]}! Please use the unbalanced OT function instead by setting omega_max!")

            else:
                print(f"Warning: Events are not balanced to within 1e-4! Using unbalanced OT with omega_max = {omega_max}!")
                
     
                s1, s2 = self.balance(s1, s2, omega_max)
                if verbose:
                    print(f"Balanced events. Time taken: {time.time() - start_time} seconds.")
                # for i in range(len(events1)):
                #     s1_, s2_ = balance(s1[i], s2[i], omega_max)

                #     s1s.append(s1_)
                #     s2s.append(s2_)
                # s1 = jnp.array(s1s)
                # s2 = jnp.array(s2s)

        # Check for empty event

        # Check batch sizes
        batch_size_1 = s1.shape[0]
        batch_size_2 = s2.shape[0]
        if batch_size_2 != batch_size_1:
            raise ValueError("Must have equal batch sizes for line-by-line sEMDs! Found batch sizes of {batch_size_1} and {batch_size_2}!")


        # Compute EMD
        emds =  self.ds2(s1, s2)

        if verbose:
            print(f"Finished: Time taken: {time.time() - start_time} seconds.")
        return emds
    

    def crossEMD(self, events1, events2, beta = 1, metric = "euclidean"):

        return self.cross_ds2_events1_events2(events1, events2)



    # Compilation handling
    def compile(self, verbose = True):
        """Function to compile the SPECTER model. This converts expressions to efficient XLA for evalulation. 
           Each function is then run once on test events to ensure that the jaxpr expressions are fully traced before compilation.
        """

        

        # ########## Compile and Vector Map Individual functions ##########

        # If verbose, keep track of compilation time:
        if verbose:
            start = time.time()

        if verbose:
            print("Compiling SPECTER model...")
            

        # Test events
        if verbose:
            print("Generating test events for tracing ...")
        test_events_1 = jnp.ones((3, 99, 3)) 
        test_events_2 = jnp.ones((3, 101, 3))
        self.compute_spectral_representation_TEMP, self.spectral_epresentation_gradients_TEMP = self.initialize_function_grad_trace(self.compute_spectral_representation, test_events_1, jacobian= True)
        self.compute_spectral_representation_spherical_TEMP, self.spectral_epresentation_spherical_gradients_TEMP = self.initialize_function_grad_trace(self.compute_spectral_representation_spherical, test_events_1, jacobian= True) 
        self.compute_spectral_representation_cylindrical_TEMP, self.spectral_epresentation_cylindrical_gradients_TEMP = self.initialize_function_grad_trace(self.compute_spectral_repsentation_cylindrical, test_events_1, jacobian= True)

        test_spectral_1 = self.compute_spectral_representation_TEMP(test_events_1)
        test_spectral_2 = self.compute_spectral_representation_TEMP(test_events_2)
    


        if verbose:
            print("Test events generated! Time taken: ", time.time() - start, " seconds.")



        # initialize ds2 functions (Need to do this first, since vmap order matters): 
        if verbose:
            print("Compiling spectral representation functions ...") 
        self.ds2, self.ds2_gradients = self.initialize_function_grad_trace(ds2, test_spectral_1, test_spectral_2)
        self.ds2_events1_events2, self.ds2_events1_events2_gradients = self.initialize_function_grad_trace(self.ds2_events1_events2, test_events_1, test_events_2)
        self.ds2_events1_spectral2, self.ds2_events1_spectral2_gradients = self.initialize_function_grad_trace(self.ds2_events1_spectral2, test_events_1, test_spectral_2)
        self.ds2_spectral1_events2, self.ds2_spectral1_events2_gradients = self.initialize_function_grad_trace(self.ds2_spectral1_events2, test_spectral_1, test_events_2)
        self.cross_ds2_events1_events2, self.cross_ds2_events1_events2_gradients = self.initialize_function_grad_trace(cross_ds2_events1_events2, test_events_1, test_events_2)

        # self.balance = jax.jit(balance)
        self.balance = jax.vmap(balance, in_axes = (0, 0, None))

        # Initialize the self.balance function
        test_output1, test_output2 = self.balance(test_spectral_1, test_spectral_2, 1.0)

        # Spectral representation function
        self.compute_spectral_representation, self.spectral_representation_gradients = self.initialize_function_grad_trace(self.compute_spectral_representation, test_events_1, jacobian= True)
        self.compute_spectral_representation_spherical, self.spectral_representation_spherical_gradients = self.initialize_function_grad_trace(self.compute_spectral_representation_spherical, test_events_1, jacobian= True)
        self.compute_spectral_repsentation_cylindrical, self.spectral_representation_cylindrical_gradients = self.initialize_function_grad_trace(self.compute_spectral_repsentation_cylindrical, test_events_1, jacobian= True)

        # ds2 function
        test_spectral_1 = self.compute_spectral_representation(test_events_1)
        test_spectral_2 = self.compute_spectral_representation(test_events_2)


           # self.ds2_events1_events2, self.ds2_events1_s2 = self.initialize_ds2_events_functions()
        # self.ds2_events1_events2 = jax.jit(self.ds2_events1_events2)
        # self.ds2_events1_s2 = jax.jit(self.ds2_events1_s2)

        # self.compiled_functions = [self.compute_spectral_representation, self.ds2, self.ds2_events1_events2, self.ds2_events1_s2]

        
        # ########## JAX Tracing ##########




        # test_ds2 = self.ds2(test_s_1, test_s_2).block_until_ready()
        # test_ds2_events1_events2 = self.ds2_events1_events2(test_events_1, test_events_2).block_until_ready()
        # test_ds2_events1_s2 = self.ds2_events1_s2(test_events_1, test_s_2).block_until_ready()

        # Initialize gradient functions for each compiled function
        # self.ds2_gradients = self.initialize_gradient_function(self.ds2)
        # self.ds2_events1_events2_gradients = self.initialize_gradient_function(self.ds2_events1_events2)
        # self.ds2_events1_s2_gradients = self.initialize_gradient_function(self.ds2_events1_s2)

        # Trace through each gradient function
        # test_ds2_gradients = self.ds2_gradients(test_s_1, test_s_2).block_until_ready()
        # test_ds2_events1_events2_gradients = self.ds2_events1_events2_gradients(test_events_1, test_events_2).block_until_ready()
        # test_ds2_events1_s2_gradients = self.ds2_events1_s2_gradients(test_events_1, test_s_2).block_until_ready()



     

        if verbose:
            print("Compilation complete! Time taken: ", time.time() - start, " seconds.")


    # ###################################### #
    # ########## HELPER FUNCTIONS ########## #
    # ###################################### #


    # Function to initialize the gradient function for a given function
    def initialize_function_grad_trace(self, function, *args, jacobian = False, vmap = True,  **kwargs):
        """Helper function to initialize the gradient function for a given function

        Args:
            function (function): Function to initialize the gradient function for

        Returns:
            function: Batchwise Gradient function for the given function
        """

        if not jacobian:
            grad = jax.vmap(jax.jit(jax.grad(function, argnums=0)))
        else:
            grad = jax.vmap(jax.jit(jax.jacfwd(function, argnums=0)))

        fun = jax.vmap(jax.jit(function))
        if not vmap:
            fun = jax.jit(function)
            grad = jax.jit(jax.grad(function, argnums=0))

        # Trace
        test = fun(*args, **kwargs).block_until_ready()
        test_grad = grad(*args, **kwargs).block_until_ready()

        return fun, grad
    


        # ds2 where both arguments are in events form
    def ds2_events1_events2(self, events1, events2):

        s1 = self.compute_spectral_representation(events1)
        s2 = self.compute_spectral_representation(events2)

        return ds2(s1, s2)
    
    # ds2 where the first argument is in events form and the second is in spectral form
    def ds2_events1_spectral2(self, events1, s2):

        s1 = self.compute_spectral_representation(events1)

        return ds2(s1, s2)
    
    # ds2 where the first argument is in spectral form and the second is in events form
    def ds2_spectral1_events2(self, s1, events2):
            
            s2 = self.compute_spectral_representation(events2)

            return ds2(s1, s2)
    
        # return ds2_events1_events2, ds2_events1_spectral2, ds2_spectral1_events2

    def cross_ds2_events1_events2(self, events1, events2):

        return cross_ds2_events1_events2(events1, events2)