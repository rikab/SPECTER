import jax
import jax.numpy as jnp
from jax import grad, jacobian, jit


from jax.example_libraries import optimizers as jax_opt
import numpy as np


# JAX Functio

# #############################################
# ########## Spectral EMD Calculator ##########
# #############################################

# @jax.jit
def weighted_sum(s, p = 2, max_index = None, inclusive = True):
    
    if max_index is None:
        return jnp.sum(s[:,:,1] * jnp.power(s[:,:,0], p), axis = -1)
    else:
        max_index = max_index + 1 if inclusive else max_index
        return jnp.sum(s[:,:max_index,1] * jnp.power(s[:,:max_index,0], p),axis = -1)


def cross_term(s1, s2):

    # Cross term
    omega1s = s1[:,:,0]
    omega2s = s2[:,:,0]

    E1s = s1[:,:,1]
    E2s = s2[:,:,1]


    E1_cumsums = jnp.cumsum(E1s, axis = -1)
    E2_cumsums = jnp.cumsum(E2s, axis = -1)
    shifted_E1_cumsums = jnp.concatenate((E1_cumsums[:,0][:,None], E1_cumsums[:,:-1]), axis = -1) 
    shifted_E2_cumsums = jnp.concatenate((E2_cumsums[:,0][:,None], E2_cumsums[:,:-1]), axis = -1) 

    
    
    omega_n_omega_l = omega1s[:,:,None] * omega2s[:,None,:]
    minE = jnp.minimum(E1_cumsums[:,:,None], E2_cumsums[:,None,:])
    maxE = jnp.maximum(shifted_E1_cumsums[:,:,None], shifted_E2_cumsums[:,None,:])
    x = minE - maxE

    cross = omega_n_omega_l * x * theta(x)
    cross_term = jnp.sum(cross, axis = (-1,-2))


    return cross_term


def theta(x):

    return x > 0 


def ds2(s1, s2):

    if pairwise == True:
        raise NotImplementedError("Pairwise sEMDs not implemented!")
    batch_size_1 = s1.shape[0]
    batch_size_2 = s2.shape[0]
    if batch_size_2 != batch_size_1 and pairwise == False:
        raise ValueError("Must have equal batch sizes for line-by-line sEMDs! Found batch sizes of {batch_size_1} and {batch_size_2}!Z")


    term1 = weighted_sum(s1)
    term2 = weighted_sum(s2)

    return term1 + term2 - 2*cross_term(s1, s2)



# ########################################
# ########## SHAPE MINIMIZATION ##########
# ########################################

# @jax.jit
def train_step(epoch, s, sprongs, return_grads = True):

        
    sEMD2s = sEMD2(sprongs, s)

    if not return_grads:
        return sEMD2

    grads = grad_sEMD2(sprongs, s)
    return sEMD2s, grads

    opt_state = opt_update(epoch, grads, opt_state)
    return opt_state, sEMDS, sprongs


# @jax.jit
def compute_2spronginess(s, alpha = 1e-3):

    batch_size = s.shape[0]
    epochs = 1000



    initial_omega = 0.5
    initial_2ee = 0.2

    # Initialize 2-prong events
    sprongs = np.zeros((batch_size, 2, 2))
    sprongs[:,0,:] = (0, 1-initial_2ee)
    sprongs[:,1,:] = (initial_omega, initial_2ee)

    # Optimizer
    opt_state = None
    opt_init, opt_update, get_params = jax_opt.adam(alpha)
    opt_state = opt_init(sprongs)

    # ts = []
    # ss = []

    for epoch in range(epochs):

        sprongs = get_params(opt_state)
        sEMDs, grads = train_step(epoch, s, sprongs)
        opt_state = opt_update(epoch, grads, opt_state)


    # plt.scatter(ts,ss)
    return sEMDs, sprongs




# @jax.jit
def sEMD2(events, s):

    sprongs = compute_spectral_representation(events)
    sEMDs = ds2(sprongs, s)

    return sEMDs

grad_sEMD2 = grad(jnp.sum(sEMD2), argnums=0)

    