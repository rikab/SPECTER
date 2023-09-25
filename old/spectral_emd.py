import numpy as np







# TODO: my SHAPER representation of events is somewhat inefficient. do over.
def compute_spectral_representation(event, pad = 125, omega_max = 2, type = "event"):

    points, zs = event[0], event[1]
    euclidean_distance_squared = np.sum(np.square(points[:, None, :] - points[None, :, :]), axis=-1)
    
    # Upper Triangle Matrices
    if type == "event":
        omega_ij = np.triu(euclidean_distance_squared / 2)
    elif type == "jet":
        omega_ij = np.triu(np.sqrt(euclidean_distance_squared ))
    ee_ij = np.triu(zs[:,None] * zs[None,:])
    ee2 = np.trace(ee_ij)

    # Flatten to 1D Spectral Representation and remove 0s
    omega_n = omega_ij.flatten()
    ee_n = 2 * ee_ij.flatten()
    mask = omega_n > 0
    
    omega_n = omega_n[mask]
    ee_n = ee_n[mask]
    s = np.stack((omega_n, ee_n)).T
    
    # Sort and append 0
    s = s[s[:,0].argsort()]
    s0 = np.array(((0, ee2,),))
    s = np.concatenate((s0, s), axis = 0)

    return s

def cumulative_spectral_function(s):
    S = s.copy()
    S[:,1] = np.cumsum(S[:,1])
    return S



def weighted_sum(s, p = 2, max_index = None, inclusive = True):

    if max_index is None:
        return np.sum(s[:,1] * np.power(s[:,0], 2))
    else:
        max_index = max_index + 1 if inclusive else max_index
        return np.sum(s[:max_index,1] * np.power(s[:max_index,0], 2))


def energy_sum(s, max_index = None, inclusive = True):
    if max_index is None:
        return np.sum(s[:,1])
    else:
        max_index = max_index + 1 if inclusive else max_index
        return np.sum(s[:max_index,1] )


def cross_term(s1, s2):

    # Cross term
    omega1s = s1[:,0]
    omega2s = s2[:,0]

    E1s = s1[:,1]
    E2s = s2[:,1]

    E1_cumsums = np.cumsum(E1s)
    E2_cumsums = np.cumsum(E2s)
    shifted_E1_cumsums = np.concatenate((np.array((E1_cumsums[0],)), E1_cumsums[:-1])) 
    shifted_E2_cumsums = np.concatenate((np.array((E2_cumsums[0],)), E2_cumsums[:-1])) 

    omega_n_omega_l = omega1s[:,None] * omega2s[None,:]
    minE = np.minimum(E1_cumsums[:,None], E2_cumsums[None,:])
    maxE = np.maximum(shifted_E1_cumsums[:,None], shifted_E2_cumsums[None,:])
    x = minE - maxE

    cross_term = np.sum(omega_n_omega_l * x * theta(x))
    return cross_term

def theta(x):

    return x > 0 


def ds2(s1, s2):

    term1 = weighted_sum(s1)
    term2 = weighted_sum(s2)

    return term1 + term2 - 2*cross_term(s1, s2)



