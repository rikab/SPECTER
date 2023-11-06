import jax
import jax.numpy as jnp
from jax import grad, jacobian, jit
from jax.lax import dynamic_slice


from jax.example_libraries import optimizers as jax_opt
import numpy as np


# ########## Spectral Representation ##########
def compute_spectral_representation(events, omega_max = 2, beta = 1, dtype = jnp.float32):
        """Function to compute the spectral representation of a set of events. Must be compiled before use on batched events -- see SPECTER.compile().

        Args:
            events (ndarray): Array of events with shape (batch_size, pad, 3)
            omega_max (float, optional): Maximum omega value. Defaults to 2.
            beta (float, optional): Beta value for the spectral representation. Defaults to 1.
            dtype (jax.numpy.dtype, optional): Data type for the output. Defaults to jax.numpy.float32.

        Returns:
            ndarray: Spectral representation of the events with shape (batch_size, pad*(pad-1)/2, 2)
        """

        # Events shape is (pad, 3)
        points, zs = events[:,1:], events[:,0]
        euclidean_distance_squared = jnp.sum(jnp.square(points[:, None, :] - points[None, :, :]), axis=-1)
        

        # Upper Triangle Matrices
        omega_ij = jnp.triu((euclidean_distance_squared), k = 1)
        triangle_indices = jnp.triu_indices(zs.shape[0], k = 1)
        triangle_indices_i = triangle_indices[0]
        triangle_indices_j = triangle_indices[1]

        # Get pairwise products of energies
        ee_ij = jnp.triu(zs[:,None] * zs[None,:])
        ee2 = jnp.trace(ee_ij, axis1 = 0, axis2=1)

        # Flatten to 1D Spectral Representation and remove 0s
        omega_n = omega_ij[triangle_indices_i, triangle_indices_j]
        omega_n = jnp.power(omega_n, beta / 2) / beta
        ee_n = 2 * ee_ij[triangle_indices_i, triangle_indices_j]

        s = jnp.stack((omega_n, ee_n), axis = 1)
        s = jnp.transpose(s, (0,1))
        
        # Sort and append 0
        indices = s[:,0].argsort()
        s = s[indices]
        s0 = jnp.zeros((1, 1))
        s1 = jnp.concatenate((s0, ee2 * jnp.ones((1,1))), axis = 1)
        s = jnp.concatenate((s1, s), axis = 0)

        

        return s.astype(dtype)



# #############################################
# ########## Spectral EMD Calculator ##########
# #############################################

def weighted_sum(s, p = 2, max_index = None, inclusive = True):
    
    if max_index is None:
        return jnp.sum(s[:,1] * jnp.power(s[:,0], p), axis = -1)
    else:
        max_index = max_index + 1 if inclusive else max_index
        return jnp.sum(s[:max_index,1] * jnp.power(s[:max_index,0], p),axis = -1)


# JAX algorithm to find index pairs (i,j) such that $X_i > Y_{j-1}$ and $Y_j > X_{i-1}$ for sorted X and Y
def find_indices_jax(X, Y):


    def condition(state):
        step, i, j, j_prev, _ = state
        return (jax.lax.lt(i, X_length) & jax.lax.lt(j, Y_length))

    def body(state):
        step, i, j, j_prev, result = state
        x_gt_y = jnp.take(X, i) >= jnp.take(Y, j - 1)
        y_gt_x = jnp.take(Y, j) >= jnp.take(X, i - 1)

        # Conditionals to update i and j
        advance_i = (x_gt_y & y_gt_x)
        advance_i_2 = ((jnp.take(X, i) <= jnp.take(Y, j-1))) & ~advance_i
        advance_j = ~advance_i & ~advance_i_2

        next_i = i + jax.lax.select(advance_i | advance_i_2, 1, 0)
        next_j = j + jax.lax.select(advance_j, 1, 0)
        next_j_prev = j_prev + jax.lax.select(advance_j, 1, 0)

        # Conditionals to decide whether to add (i,j) or null = (-1,-1)
        next_result = jax.lax.cond(
            x_gt_y & y_gt_x,
            lambda _: jnp.array([i, j], dtype=jnp.int32),
            lambda _: jnp.array([-1, -1], dtype=jnp.int32),
            None
        )

        next_result = result.at[step].set(next_result)
        return (step + 1, next_i, next_j, next_j_prev, next_result)
    
    # Initializations
    step = 0
    i_init = 1
    j_init = 1
    j_prev_init = 1
    X_length, Y_length = X.shape[-1], Y.shape[-1]
    max_length = X_length + Y_length
    initial_state = (step, i_init, j_init, j_prev_init, jnp.full((max_length, 2), -1, dtype=jnp.int32), )

    # Run while loop
    _, _, _, _, final_result = jax.lax.while_loop(condition, body, initial_state)

    return final_result


# Function to find indices on X and Y, then Y and X, and combine:
def find_indices(X, Y):
        
        x_size = X.shape[0]
        y_size = Y.shape[0]
    
        # Find indices on X and Y
        indices = find_indices_jax(X, Y)
        i_indices, j_indices = indices[:,0], indices[:,1]
    
        # Find indices on Y and X
        indices = find_indices_jax(Y, X)
        j_indices_2, i_indices_2 = indices[:,0], indices[:,1]
    
        # Combine
        i_indices = jnp.concatenate((i_indices, i_indices_2))
        j_indices = jnp.concatenate((j_indices, j_indices_2))

        # Remove duplicates
        indices = jnp.unique(jnp.stack((i_indices, j_indices), axis = -1), size = 2 * (x_size + y_size), fill_value = -1, axis = 0)
        mask = (indices[:,0] > -1) * (indices[:,1] > -1)

        return indices, mask


# Compute the cross term using the OLD parralized algorithm [DEPRECATED]
def cross_term(s1, s2):

    # Cross term
    omega1s = s1[:,0]
    omega2s = s2[:,0]

    E1s = s1[:,1]
    E2s = s2[:,1]


    E1_cumsums = jnp.cumsum(E1s, axis = -1)
    E2_cumsums = jnp.cumsum(E2s, axis = -1)
    shifted_E1_cumsums = jnp.concatenate((jnp.array((E1_cumsums[0],)), E1_cumsums[:-1]), axis = -1) 
    shifted_E2_cumsums = jnp.concatenate((jnp.array((E2_cumsums[0],)), E2_cumsums[:-1]), axis = -1) 

    
    
    omega_n_omega_l = omega1s[:,None] * omega2s[None,:]
    minE = jnp.minimum(E1_cumsums[:,None], E2_cumsums[None,:])
    maxE = jnp.maximum(shifted_E1_cumsums[:,None], shifted_E2_cumsums[None,:])
    x = minE - maxE

    cross = omega_n_omega_l * x * theta(x)
    cross_term = jnp.sum(cross, axis = (-1,-2))


    return cross_term

def sub_cross_term(omega1s, omega2, E1_cumsums, E2_cumsum, shifted_E1_cumsums, shifted_E2_cumsum):

    omega_n_omega_l = omega1s * omega2
    minE = jnp.minimum(E1_cumsums, E2_cumsum)
    maxE = jnp.maximum(shifted_E1_cumsums, shifted_E2_cumsum)
    x = minE - maxE

    cross = omega_n_omega_l * x * theta(x)
    cross_term = jnp.sum(cross, axis = -1)
    return cross_term

vmapped_sub_cross_term = jax.vmap(sub_cross_term, in_axes = (None, 0, None, 0, None, 0), out_axes = 0)


def cross_term_2_loop(index, vals):
     
    result, omega1s, omega2s, E1_cumsums, E2_cumsums, shifted_E1_cumsums, shifted_E2_cumsums, prev_index = vals

    omega_n_omega_l = omega1s * omega2s[index]
    minE = jnp.minimum(E1_cumsums, E2_cumsums[index])
    maxE = jnp.maximum(shifted_E1_cumsums, shifted_E2_cumsums[index])
    x = minE - maxE

    t = theta(x)
    prev_index = jnp.argmax(t)
    cross = omega_n_omega_l * x * theta(x)
    result += jnp.sum(cross, axis = -1)

    return result, omega1s, omega2s, E1_cumsums, E2_cumsums, shifted_E1_cumsums, shifted_E2_cumsums, prev_index     
     

# @jax.checkpoint
def cross_term_2_scan(vals, index):
    result, omega1s, omega2s, E1_cumsums, E2_cumsums, shifted_E1_cumsums, shifted_E2_cumsums, prev_index = vals

    omega_n_omega_l = omega1s * omega2s[index]
    minE = jnp.minimum(E1_cumsums, E2_cumsums[index])
    maxE = jnp.maximum(shifted_E1_cumsums, shifted_E2_cumsums[index])
    x = minE - maxE

    t = theta(x)
    prev_index = jnp.argmax(t)
    cross = omega_n_omega_l * x * t
    result += jnp.sum(cross, axis=-1)

    # Return a dummy value (None)
    return (result, omega1s, omega2s, E1_cumsums, E2_cumsums, shifted_E1_cumsums, shifted_E2_cumsums, prev_index), None


# Compute the cross term using the OLD parralized algorithm [DEPRECATED]
def cross_term_2(s1, s2):

    # Cross term
    omega1s = s1[:,0]
    omega2s = s2[:,0]

    E1s = s1[:,1]
    E2s = s2[:,1]


    E1_cumsums = jnp.cumsum(E1s, axis = -1)
    E2_cumsums = jnp.cumsum(E2s, axis = -1)
    shifted_E1_cumsums = jnp.concatenate((jnp.array((E1_cumsums[0],)), E1_cumsums[:-1]), axis = -1) 
    shifted_E2_cumsums = jnp.concatenate((jnp.array((E2_cumsums[0],)), E2_cumsums[:-1]), axis = -1) 

    # result = 0
    # def temp(result, omega1s, omega2s, E1_cumsums, E2_cumsums, shifted_E1_cumsums, shifted_E2_cumsums):
    #     return result + jnp.sum(vmapped_sub_cross_term(omega1s, omega2s, E1_cumsums, E2_cumsums, shifted_E1_cumsums, shifted_E2_cumsums), axis = -1)

    # result = temp(result, omega1s, omega2s, E1_cumsums, E2_cumsums, shifted_E1_cumsums, shifted_E2_cumsums)

    # # For loop over the second event
    # vals = (0, omega1s, omega2s, E1_cumsums, E2_cumsums, shifted_E1_cumsums, shifted_E2_cumsums, 0)
    # result, _, _, _, _, _, _, _ = jax.lax.fori_loop(0, omega2s.shape[0], cross_term_2_loop, vals)


    # Use lax.scan to perform the loop
    vals = (0, omega1s, omega2s, E1_cumsums, E2_cumsums, shifted_E1_cumsums, shifted_E2_cumsums, 0)
    vals, _ = jax.lax.scan(cross_term_2_scan, vals, jnp.arange(omega2s.shape[0]))
    result = vals[0]

    return result

# Calculate the cross term using the improved JAX finding finding algorithm ~ O(n^2)
def cross_term_improved(s1, s2):

    # Cross term
    omega1s = s1[:,0]
    omega2s = s2[:,0]

    E1s = s1[:,1]
    E2s = s2[:,1]

    # Calculate cumulative spectral functions
    cumulative_inclusive_1 = jnp.cumsum(E1s, axis = -1)
    cumulative_inclusive_2 = jnp.cumsum(E2s, axis = -1)

    Etot1 = jnp.sum(E1s)
    Etot2 = jnp.sum(E2s)

    # Exclusive = inclusive - 2EE
    cumulative_exclusive_1 = cumulative_inclusive_1 - E1s
    cumulative_exclusive_2 = cumulative_inclusive_2 - E2s

    # Determine which indices survive the theta function O(n^2)
    indices, mask = find_indices(cumulative_inclusive_1, cumulative_inclusive_2)
    indices = jax.lax.stop_gradient(indices)
    mask = jax.lax.stop_gradient(mask)
    i_indices, j_indices = indices[:,0], indices[:,1]

    
    # O(n2) parts -- calculate the cross term using the nonzero indices
    omega_n_omega_l = omega1s[i_indices] * omega2s[j_indices]
    minE = -jnp.maximum(Etot1 - cumulative_inclusive_1[i_indices], Etot2 - cumulative_inclusive_2[j_indices]) + jnp.minimum(Etot1 - cumulative_exclusive_1[i_indices], Etot2 -  cumulative_exclusive_2[j_indices])

    # Sum over the nonzero indices
    cross = omega_n_omega_l * minE
    cross_term = jnp.sum(cross * mask, axis = (-1))


    return cross_term


def theta(x):

    return jnp.heaviside(x, 0.0) 

def ds2(s1, s2):

    term1 = weighted_sum(s1)
    term2 = weighted_sum(s2)

    return term1 + term2 - 2*cross_term_2(s1, s2)


# ########## ds2 variants ##########

def ds2_events1_events2(events1, events2):
    """Function to compute the spectral Earth Mover's Distance between two sets of events

    Args:
        events1 (ndarray): Array of events with shape (batch_size, pad1, 3)
        events2 (ndarray): Array of events with shape (batch_size, pad2, 3)

    Returns:
        ndarray: Spectral EMD between events1 and events2 of size (batch_size,)
    """

    s1 = compute_spectral_representation(events1)
    s2 = compute_spectral_representation(events2)

    return ds2(s1, s2)

# ds2 where the first argument is in events form and the second is in spectral form
def ds2_events1_spectral2(events1, s2):
    """Function to compute the spectral Earth Mover's Distance between two sets of events

    Args:
        events1 (ndarray): Array of events with shape (batch_size, pad1, 3)
        s2 (ndarray): Array of spectral events with shape (batch_size, pad2*(pad2-1)/2, 2)

    Returns:
        ndarray: Spectral EMD between events1 and events2 of size (batch_size,)
    """

    s1 = compute_spectral_representation(events1)

    return ds2(s1, s2)

# ds2 where the first argument is in spectral form and the second is in events form
def ds2_spectral1_events2(s1, events2):
    """Function to compute the spectral Earth Mover's Distance between two sets of events


    Args:
        s1 (ndarray): Array of spectral events with shape (batch_size, pad1*(pad1-1)/2, 2)
        events2 (ndarray): Array of events with shape (batch_size, pad2, 3)

    Returns:
        ndarray: Spectral EMD between events1 and events2 of size (batch_size,)
    """
        
    s2 = compute_spectral_representation(events2)

    return ds2(s1, s2)


# ###########################
# ########## TESTS ##########
# ###########################




# ########################################
# ########## SHAPE MINIMIZATION ##########
# ########################################

# # @jax.jit
# def train_step(epoch, s, sprongs, return_grads = True):

        
#     sEMD2s = sEMD2(sprongs, s)

#     if not return_grads:
#         return sEMD2

#     grads = grad_sEMD2(sprongs, s)
#     return sEMD2s, grads

#     opt_state = opt_update(epoch, grads, opt_state)
#     return opt_state, sEMDS, sprongs


# # @jax.jit
# def compute_2spronginess(s, alpha = 1e-3):

#     batch_size = s.shape[0]
#     epochs = 1000



#     initial_omega = 0.5
#     initial_2ee = 0.2

#     # Initialize 2-prong events
#     sprongs = np.zeros((batch_size, 2, 2))
#     sprongs[:,0,:] = (0, 1-initial_2ee)
#     sprongs[:,1,:] = (initial_omega, initial_2ee)

#     # Optimizer
#     opt_state = None
#     opt_init, opt_update, get_params = jax_opt.adam(alpha)
#     opt_state = opt_init(sprongs)

#     # ts = []
#     # ss = []

#     for epoch in range(epochs):

#         sprongs = get_params(opt_state)
#         sEMDs, grads = train_step(epoch, s, sprongs)
#         opt_state = opt_update(epoch, grads, opt_state)


#     # plt.scatter(ts,ss)
#     return sEMDs, sprongs




# # @jax.jit
# def sEMD2(events, s):

#     sprongs = compute_spectral_representation(events)
#     sEMDs = ds2(sprongs, s)

#     return sEMDs

# grad_sEMD2 = grad(jnp.sum(sEMD2), argnums=0)

    