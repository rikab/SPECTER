import jax
import jax.numpy as jnp
from jax import grad, jacobian, jit
from jax.lax import dynamic_slice


from jax.example_libraries import optimizers as jax_opt
import numpy as np




def euclidean_metric(points):
    return jnp.sum(jnp.square(points[:, None, :] - points[None, :, :]), axis=-1)



def euclidean_metric_cross(points1, points2):
    return jnp.sum(jnp.square(points1[:, None, :] - points2[None, :, :]), axis=-1)


def cylindrical_metric(points):

    ETA_INDEX = 0
    PHI_INDEX = 1
    delta_eta = jnp.square(points[:, None, ETA_INDEX] - points[None, :, ETA_INDEX])

    # Make phi periodic by using the periodicity of the cosine
    delta_phi = jnp.square(jnp.arccos(jnp.cos(points[:, None, PHI_INDEX] - points[None, :, PHI_INDEX])))
    
    return delta_eta + delta_phi


def spherical_metric(points):

    # Square so the beta = 1 default works
    #
    return jnp.nan_to_num(jnp.square(1 - (jnp.sum(points[:, None, :] * points[None, :, :], axis=-1) / jnp.linalg.norm(points[:, None, :], axis=-1) / jnp.linalg.norm(points[None, :], axis=-1))))



    return jnp.nan_to_num(jnp.square(jnp.arccos(jnp.sum(points[:, None, :] * points[None, :, :], axis=-1) / jnp.linalg.norm(points[:, None, :], axis=-1) / jnp.linalg.norm(points[None, :], axis=-1))))
    return jnp.sqrt(1 - jnp.sum(points[:, None, :] * points[None, :, :], axis=-1) / (jnp.linalg.norm(points[:, None, :], axis=-1) * jnp.linalg.norm(points[None, :, :], axis=-1)))


def spherical_metric_cross(points1, points2):
    
    return jnp.nan_to_num(jnp.square(1 - (jnp.sum(points1[:, None, :] * points2[None, :, :], axis=-1) / jnp.linalg.norm(points1[:, None, :], axis=-1) / jnp.linalg.norm(points2[None, :], axis=-1))))
    return jnp.nan_to_num(jnp.square(jnp.arccos(jnp.sum(points1[:, None, :] * points2[None, :, :], axis=-1) / jnp.linalg.norm(points1[:, None, :], axis=-1) / jnp.linalg.norm(points2[None, :], axis=-1))))
    return jnp.sqrt(1 - jnp.sum(points1[:, None, :] * points2[None, :, :], axis=-1) / (jnp.linalg.norm(points1[:, None, :], axis=-1) * jnp.linalg.norm(points2[None, :, :], axis=-1)))


# ########## Spectral Representation ##########
def compute_spectral_representation(events, omega_max = 2, beta = 1.0, dtype = jnp.float32, euclidean = True):
        """Function to compute the spectral representation of a set of events. Must be compiled before use on batched events -- see SPECTER.compile().
euclidean_distance_squared
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


        euclidean_distance_squared = jax.lax.cond(euclidean, euclidean_metric, spherical_metric, points)


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

        # Set omega to 0 when energy is 0
        omega_n = jnp.where(ee_n == 0, 0, omega_n)

        s = jnp.stack((omega_n, ee_n), axis = 1)
        s = jnp.transpose(s, (0,1))
        
        # Sort and append 0
        indices = s[:,0].argsort()
        s = s[indices]
        s0 = jnp.zeros((1, 1))
        s1 = jnp.concatenate((s0, ee2 * jnp.ones((1,1))), axis = 1)
        s = jnp.concatenate((s1, s), axis = 0)

        

        return s.astype(dtype)




# ########## Spectral Representation ##########
def compute_spectral_representation_cylindrical(events, omega_max = 2, beta = 1.0, dtype = jnp.float32):
        """Function to compute the spectral representation of a set of events. Must be compiled before use on batched events -- see SPECTER.compile().
euclidean_distance_squared
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


        euclidean_distance_squared = cylindrical_metric(points)


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

        # Set omega to 0 when energy is 0
        omega_n = jnp.where(ee_n == 0, 0, omega_n)

        s = jnp.stack((omega_n, ee_n), axis = 1)
        s = jnp.transpose(s, (0,1))
        
        # Sort and append 0
        indices = s[:,0].argsort()
        s = s[indices]
        s0 = jnp.zeros((1, 1))
        s1 = jnp.concatenate((s0, ee2 * jnp.ones((1,1))), axis = 1)
        s = jnp.concatenate((s1, s), axis = 0)

        

        return s.astype(dtype)





def compute_double_spectral_representation(event1, event2, omega_max = 2, beta = 1.0, dtype = jnp.float32, euclidean = True):

    s1 = compute_spectral_representation(event1, omega_max, beta, dtype, euclidean)
    s2 = compute_spectral_representation(event2, omega_max, beta, dtype, euclidean)


    # concatenate the two spectral representations, sort by omega
    s = jnp.concatenate((s1, s2), axis = 0)
    indices = s[:,0].argsort()
    s = s[indices]

    return s



def compute_cross_spectral_representation(event1, event2, omega_max = 2, beta = 1.0, dtype = jnp.float32, euclidean = True):

    points1, zs1 = event1[:,1:], event1[:,0]
    points2, zs2 = event2[:,1:], event2[:,0]

    euclidean_distance_squared = jax.lax.cond(euclidean, euclidean_metric_cross, spherical_metric_cross, points1, points2)
    e_ij = 2 * zs1[:,None] * zs2[None,:]


    omega_n = euclidean_distance_squared.flatten()
    omega_n = jnp.power(omega_n, beta / 2) / beta
    ee_n = e_ij.flatten()

    s = jnp.stack((omega_n, ee_n), axis = 1)
    s = jnp.transpose(s, (0,1))

    # Sort
    indices = s[:,0].argsort()
    s = s[indices]

    return s.astype(dtype)

    
def balance(s1, s2, omega):

    total_energy1 = jnp.sum(s1[:,1])
    total_energy2 = jnp.sum(s2[:,1])

    difference = total_energy1 - total_energy2

    # Determine which event has less energy
    if total_energy1 < total_energy2:
        # s1 has less energy, so we need to add energy to it
        s1 = jnp.concatenate((s1, jnp.array([[omega, -difference]])), axis=0)

        # Resorting the combined array by omega
        s1 = jnp.sort(s1, axis=0)

        # add a blank entry to s2 to balance the arrays
        s2 = jnp.concatenate((s2, jnp.array([[0, 0]])), axis=0)
        s2 = jnp.sort(s2, axis=0)

    else:
        # s2 has less energy, so we need to add energy to it
        s2 = jnp.concatenate((s2, jnp.array([[omega, difference]])), axis=0)

        # Resorting the combined array by omega
        s2 = jnp.sort(s2, axis=0)

        # add a blank entry to s1 to balance the arrays
        s1 = jnp.concatenate((s1, jnp.array([[0, 0]])), axis=0)
        s1 = jnp.sort(s1, axis=0)



    return s1, s2



# #############################################
# ########## Spectral EMD Calculator ##########
# #############################################

def weighted_sum(s, p = 2, max_index = None, inclusive = True):
    
    if max_index is None:
        return jnp.sum(s[:,1] * jnp.power(s[:,0], p), axis = -1)
    else:
        max_index = max_index + 1 if inclusive else max_index
        return jnp.sum(s[:max_index,1] * jnp.power(s[:max_index,0], p),axis = -1)


def find_indices_jax(X, Y):

    pairs = jnp.zeros((X.shape[0] + Y.shape[0], 2), dtype=int)

    j_indices = jnp.searchsorted(Y, X, side = 'left')
    i_indices = jnp.searchsorted(X, Y, side = 'left')

    # Update the 'pairs' array using the new 'at' method
    pairs = pairs.at[0:X.shape[0], 0].set(jnp.arange(X.shape[0]))
    pairs = pairs.at[0:X.shape[0], 1].set(j_indices)
    pairs = pairs.at[X.shape[0]:X.shape[0] + Y.shape[0], 0].set(i_indices)
    pairs = pairs.at[X.shape[0]:X.shape[0] + Y.shape[0], 1].set(jnp.arange(Y.shape[0]))

    return pairs


# Function to find indices on X and Y, then Y and X, and combine:
def find_indices(X, Y):
        
        x_size = X.shape[0]
        y_size = Y.shape[0]
    
        # Find indices on X and Y
        indices = find_indices_jax(X, Y)

        # Remove duplicates
        indices = jnp.unique(indices, size = (x_size + y_size), fill_value = -1, axis = 0)
        mask = (indices[:,0] > 0) * (indices[:,1] > 0) * (indices[:,0] < X.shape[0]) * (indices[:,1] < Y.shape[0])

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
    cumulative_exclusive_1 = jnp.concatenate((cumulative_inclusive_1[:1],
                                          cumulative_inclusive_1[:-1]))
    cumulative_exclusive_2 = jnp.concatenate((cumulative_inclusive_2[:1],
                                          cumulative_inclusive_2[:-1]))

    # Determine which indices survive the theta function O(n^2)
    indices, mask = find_indices(cumulative_inclusive_1, cumulative_inclusive_2)
    # indices = jax.lax.stop_gradient(indices)
    # mask = jax.lax.stop_gradient(mask)
    i_indices, j_indices = indices[:,0], indices[:,1]

    
    # O(n2) parts -- calculate the cross term using the nonzero indices
    omega_n_omega_l = omega1s[i_indices] * omega2s[j_indices]
    minE = -jnp.maximum(Etot1 - cumulative_inclusive_1[i_indices], Etot2 - cumulative_inclusive_2[j_indices]) + jnp.minimum(Etot1 - cumulative_exclusive_1[i_indices], Etot2 -  cumulative_exclusive_2[j_indices])
    cross = omega_n_omega_l * minE

    # Get a mask of the places where omega_n_omega_l is nonzero
    nonzero = omega_n_omega_l > 0
    nonzero_cross = jnp.where(nonzero, cross, 0)
    nonzero_mask = jnp.where(mask, nonzero, 0)



    cross_term = jnp.sum(nonzero_cross * nonzero_mask, axis = (-1))


    return cross_term


def theta(x):

    return jnp.heaviside(x, 0.0) 

def ds2(s1, s2):

    term1 = weighted_sum(s1)
    term2 = weighted_sum(s2)

    return term1 + term2 - 2*cross_term_improved(s1, s2)


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


def cross_ds2_events1_events2(events1, events2):
    """Function to compute the spectral Earth Mover's Distance between two sets of events

    Args:
        events1 (ndarray): Array of events with shape (batch_size, pad1, 3)
        events2 (ndarray): Array of events with shape (batch_size, pad2, 3)

    Returns:
        ndarray: Spectral EMD between events1 and events2 of size (batch_size,)
    """

    s1 = compute_double_spectral_representation(events1, events2)
    s2 = compute_cross_spectral_representation(events1, events2)

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

    