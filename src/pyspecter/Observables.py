
# Standard Imports
import numpy as np
from time import time
from matplotlib import pyplot as plt
import os


# Jax
from jax import grad, jacobian, jit, checkpoint, vmap, tree_map, value_and_grad
import jax.lax as lax
import jax.numpy as jnp
from jax import random
import jax.example_libraries.optimizers as jax_opt



from pyspecter.SpectralEMD_Helper import ds2_events1_spectral2, compute_spectral_representation



class Observable():

    def __init__(self, sampler, name, initializer = None, projector = None, is_trivial = False):

        self.sampler = sampler
        self.name = name

        if initializer is not None:
            self.initializer = initializer
        else:
            self.initializer = lambda: None

        if projector is not None:
            self.projector = projector
        else:
            self.projector = lambda x: x

        self.is_trivial = is_trivial
        self.skip_vmap_initialization = False
        # self.is_trivial = (len(list(self.param_dict.keys())) == 0)


    # Compile the training loop, along with vmaps and gradients
    def compile(self, test_events = None, vmap_before_jit = False, **kwargs):

        # Get the gradient of the spectral EMDs
        self.gradient_train_step = grad(self.train_step, argnums = 2)
        self.gradient_and_loss = value_and_grad(self.train_step, argnums = 2)

        if  vmap_before_jit:

            print("Compiling with vmap before jit")

            # vmap everything
            vmapped_compute_spectral_representation = vmap(compute_spectral_representation, in_axes = (0,))
            vmapped_train_step = vmap(self.train_step, in_axes = (None, 0, 0, None))

            if self.skip_vmap_initialization:
                vmapped_initializer = self.initializer
            else:
                vmapped_initializer = vmap(self.initializer, in_axes = (0, None, None))
            vmapped_projector = vmap(self.projector, in_axes = (0,))
            vmapped_gradient_train_step = vmap(self.gradient_train_step, in_axes = (None, 0, 0, None))
            vmapped_finite_differences_gradient = vmap(self.finite_differences_gradient, in_axes = (None, 0, 0, 0, None))
            vmapped_gradient_and_loss = vmap(self.gradient_and_loss, in_axes = (None, 0, 0, None))


            # Compile everything
            self.compiled_compute_spectral_representation = jit(compute_spectral_representation)
            self.vmapped_compute_spectral_representation = jit(vmapped_compute_spectral_representation)
            self.vmapped_train_step = jit(vmapped_train_step, static_argnums= (3,))
            if self.skip_vmap_initialization:
                self.vmapped_initializer = self.initializer
            else:
                self.vmapped_gradient_train_step = jit(vmapped_gradient_train_step, static_argnums= (3,))
            self.vmapped_initializer = jit(vmapped_initializer)
            self.vmapped_projector = jit(vmapped_projector)
            self.vmapped_finite_differences_gradient = jit(vmapped_finite_differences_gradient, static_argnums=(4,))
            self.vmapped_gradient_and_loss = jit(vmapped_gradient_and_loss, static_argnums=(3,))

        else:

            print("Compiling with vmap after jit")

            # Compile everything
            jit_compute_spectral_representation = jit(compute_spectral_representation)
            jit_train_step = jit(self.train_step, static_argnums=(3,))
            jit_gradient_train_step = jit(self.gradient_train_step, static_argnums=(3,))
            jit_initializer = jit(self.initializer)
            jit_projector = jit(self.projector)
            jit_finite_differences_gradient = jit(self.finite_differences_gradient, static_argnums=(4,))
            jit_gradient_and_loss = jit(self.gradient_and_loss, static_argnums=(3,))

            # vmapped everything
            self.vmapped_compute_spectral_representation = vmap(jit_compute_spectral_representation, in_axes = (0,))
            self.vmapped_train_step = vmap(jit_train_step, in_axes = (None, 0, 0, None))
            self.vmapped_gradient_train_step = vmap(jit_gradient_train_step, in_axes = (None, 0, 0, None))
            
            if self.skip_vmap_initialization:
                self.vmapped_initializer = self.initializer
            else:
                self.vmapped_initializer = vmap(jit_initializer, in_axes = (0, None, None))
            self.vmapped_projector = vmap(jit_projector, in_axes = (0,))
            self.vmapped_finite_differences_gradient = vmap(jit_finite_differences_gradient, in_axes = (None, 0, 0, 0, None))
            self.vmapped_gradient_and_loss = vmap(jit_gradient_and_loss, in_axes = (None, 0, 0, None))



        if test_events is not None:
            x, y, z, w = self.compute(test_events, verbose=False, **kwargs)
        
    def compute_single_event(self, event, learning_rate = 0.001, epochs = 150, N_sample = 25, finite_difference = False, seed = 0, verbose = True):

        # Initialize
        spectral_event = self.compiled_compute_spectral_representation(event)
        params =   self.initializer(event, N_sample, seed)

        # Optimizer
        opt_state = None
        opt_init, opt_update, get_params = jax_opt.adam(learning_rate)
        opt_state = opt_init(params)


        losses = np.zeros((epochs,))
        param_history = []

        # If observable is trivial, just do a single iteration
        if self.is_trivial:
            sEMD = self.train_step(0, spectral_event, params, N_sample)
            return sEMD, params, losses
        
        for epoch in range(epochs):

            params = get_params(opt_state)
            params = self.projector(params)
            param_history.append(params)
            
            if finite_difference:
                sEMD = self.train_step(epoch, spectral_event, params, N_sample)
                grads = self.finite_differences_gradient(epoch, sEMD, spectral_event, params, N_sample)
            else:
                sEMD, grads = self.gradient_and_loss(epoch, spectral_event, params, N_sample, seed = seed)
            
            # Fix NaNs
            for key in grads.keys():
                grads[key] = jnp.nan_to_num(grads[key])
            
            opt_state = opt_update(epoch, grads, opt_state)

            # Apply the separate function to modify the parameters
            new_params = self.projector(get_params(opt_state))

            # Manually modify the opt_state's parameters without resetting internal state
            opt_state = replace_params_in_state(opt_state, new_params)
            losses[epoch] = sEMD

            if verbose:
                print(f"{self.name}: Epoch {epoch}, Loss: {sEMD}")

        # Get the epoch with the lowest loss
        best_epoch = np.argmin(losses)
        best_params = param_history[best_epoch]
        return losses[best_epoch], best_params, losses, param_history
    
    def compute(self, events, learning_rate = 0.001, epochs = 150, early_stopping = 10, early_stopping_fraction = 0.95, N_sample = 25, finite_difference = False, seed = 0, verbose = True):

        # Initialize
        spectral_event = self.vmapped_compute_spectral_representation(events)
        params =   self.vmapped_initializer(events, N_sample, seed)
        start_time = time()
        time_history = [start_time]

        # Optimizer
        opt_state = None
        opt_init, opt_update, get_params = jax_opt.adam(learning_rate)
        opt_state = opt_init(params)

        losses = np.ones((epochs,events.shape[0])) * 99999
        early_stopping_counter = np.zeros((events.shape[0],), dtype = np.int32)
        early_stopping_mask = jnp.ones((events.shape[0],), dtype = np.bool)
        is_done = jnp.ones((events.shape[0],), dtype = np.bool)
        best_params = params.copy()
        params_history = []

        for epoch in range(epochs):

            params = get_params(opt_state)
            params = self.vmapped_projector(params)
            params_history.append(params)

            masked_spectral_events = spectral_event[early_stopping_mask]
            masked_params = mask_dict(params, early_stopping_mask)

            
            if  finite_difference:
                sEMD = self.vmapped_train_step(epoch, masked_spectral_events, masked_params, N_sample)
                masked_grads = self.vmapped_finite_differences_gradient(epoch, sEMD, masked_spectral_events, masked_params, N_sample)
            else:
                sEMD, masked_grads = self.vmapped_gradient_and_loss(epoch, masked_spectral_events, masked_params, N_sample)
            
            # Unmask the gradients and sEMD
            grads = {k: jnp.zeros_like(v) for k, v in params.items()}
            for key in grads.keys():
                grads[key] = grads[key].at[early_stopping_mask].set(masked_grads[key])
            unmasked_sEMD = jnp.zeros_like(losses[epoch])
            unmasked_sEMD = unmasked_sEMD.at[early_stopping_mask].set(sEMD)

            # Fix NaNs
            for key in grads.keys():
                grads[key] = jnp.nan_to_num(grads[key])

            # Gradient update
            opt_state = opt_update(epoch, grads, opt_state)

           # Apply the separate function to modify the parameters
            new_params = self.vmapped_projector(get_params(opt_state))

            # Manually modify the opt_state's parameters without resetting internal state
            opt_state = replace_params_in_state(opt_state, new_params)
            losses[epoch] = jnp.where(early_stopping_mask, unmasked_sEMD, losses[epoch-1])

            # if the loss has not changed in 10 epochs, stop
            early_stopping_epoch = max(epoch - early_stopping, 0)

            if epoch >= 1:
                mins = jnp.min(losses[early_stopping_epoch:epoch], axis=0)
                # early_stopping_mask = early_stopping_mask & (early_stopping_counter < early_stopping)
                is_done = is_done & (early_stopping_counter < early_stopping)
                losses[epoch] = jnp.where(early_stopping_mask, unmasked_sEMD, mins)
                early_stopping_counter = jnp.where(losses[epoch] >= mins, early_stopping_counter + 1, 0)
                

                # Update best_params for events where loss has decreased
                update_mask = unmasked_sEMD < losses[epoch-1]
                for key in params.keys():

                    # add None dimensions to update_mask to match the shape of new_params[key]
                    mask_broadcasted = lax.broadcast_in_dim(update_mask, best_params[key].shape, broadcast_dimensions=(0,))
                    best_params[key] = lax.select(mask_broadcasted, new_params[key], best_params[key])
                    
            frac = 1 - np.sum(is_done)/events.shape[0]

            if verbose:
                current_time = time()
                time_history.append(current_time)
                print(f"{self.name}: Epoch {epoch} of {epochs}, Mean Loss: {jnp.mean(losses[epoch]) : .3e}, Time: {current_time - start_time : .3f}s ({time_history[-1] - time_history[-2] : .3f}s), Early Stopping: {frac : .3f}")


            if np.all(~early_stopping_mask) or frac > early_stopping_fraction:
                break     

        return jnp.min(losses, axis = 0), best_params, losses, params_history



    def sample(self, params, n_samples, seed):

        return self.sampler(params, n_samples, seed)
    

    def project(self, params):

        return self.projector(params)


    def train_step(self, epoch, spectral_event, params, sample_N, seed = 0):

        shape_event = self.sample(params, sample_N, seed = epoch + seed)
        sEMDS = checkpoint(ds2_events1_spectral2)(shape_event, spectral_event)
        return sEMDS
    

    def finite_differences_gradient(self, epoch, sEMD, spectral_event, params, N_sample = 25, epsilon=1e-2):
        """
        Compute the gradient of `loss_fn` with respect to `params` using finite differences.
        This version uses multiplicative epsilon and is JAX-compilable.
        
        Args:
            params (dict): A dictionary containing the parameters.
            loss_fn (callable): The loss function to compute the gradient of.
            x (array): The input data for the loss function.
            y (array): The output data for the loss function.
            epsilon (float, optional): The small relative change to apply to each parameter to calculate finite differences.
        
        Returns:
            dict: A dictionary of gradients for each parameter.
        """
    
        def get_perturbed_loss(delta):
            # Perturb each parameter by a small relative amount and evaluate the loss
            perturbed_params = tree_map(lambda v: v * (1 + delta) + 1e-4, params)
            return self.train_step(epoch, spectral_event, perturbed_params, N_sample)

        # Compute the perturbed losses for positive and negative epsilon
        loss_plus_epsilon = get_perturbed_loss(epsilon)
        # loss_minus_epsilon = get_perturbed_loss(-epsilon)

        # Use jax.tree_multimap to compute the gradient for each parameter
        gradients = tree_map(
            lambda v: (loss_plus_epsilon - sEMD) / (epsilon * v + 1e-4),
            params
        )
    
        return gradients
    




def mask_dict(d, mask):
    return {k: v[mask] for k, v in d.items()}


def replace_params_in_state(opt_state, new_params):
    if isinstance(opt_state, tuple) and len(opt_state) == 2 and isinstance(opt_state[0], dict):
        # This is the parameter tuple for Adam
        return (new_params, opt_state[1])
    elif isinstance(opt_state, tuple):
        # Unpack and modify recursively
        return tuple(replace_params_in_state(sub_state, new_params) for sub_state in opt_state)
    else:
        # Leaf node or unknown type, return unchanged
        return opt_state

