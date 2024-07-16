import jax.numpy as jnp
from jax import random
from jax import grad, jacobian, jit


from pyspecter.Observables import Observable
from pyspecter.utils.data_utils import kT_N




def build_jet_observables(R = 0.5):


    observables_dict = {}

    # %%%%%%%%%% LINE %%%%%%%%%%

    def sample_line(params, N, seed):

        key = random.PRNGKey(seed)
        ones = random.uniform(key, shape=(N,), minval=0., maxval=1)
        
        # Compute x and y coordinates of the sampled points on the line
        x = params["Length"] * ones
        y = params["Length"] * jnp.zeros_like(ones)
        
        event = jnp.column_stack([jnp.ones(N) / N, x, y])
        
        return event 

    def initialize_line(event, N, seed):
        return {"Length" : R}

    def project_line(params):

        radius = params["Length"]
        params["Length"] = jnp.maximum(radius, 0.)
        return params
    
    def hard_compute_line(spectral_event):

        omegas = spectral_event[:,0]
        two_EE = spectral_event[:,1]
        E_tot = jnp.sum(two_EE)

        # Cumulative spectral function, inclusive and exclusive
        cumulative_spectral_function = jnp.cumsum(two_EE)
        cumulative_exclusive = cumulative_spectral_function - two_EE

        term0 = two_EE
        term1 = 2/3 / E_tot * jnp.power(E_tot - cumulative_spectral_function, 3/2)
        term2 = 2/3 / E_tot * jnp.power(E_tot - cumulative_exclusive, 3/2)

        l_opt = 6 * jnp.sum(omegas * (term0 + term1 - term2)) / E_tot**2
        semd_opt = jnp.sum(omegas**2 * two_EE) -   l_opt**2 /6

        return semd_opt, l_opt


    _splineliness = Observable(sample_line, name = "spLineliness", initializer=initialize_line, projector=project_line, hard_compute_function = hard_compute_line)
    _splineliness.compile()

    # %%%%%%%%%% RING %%%%%%%%%%

    def sample_circle(params, N, seed):

        key = random.PRNGKey(seed)
        thetas = random.uniform(key, shape=(N,), minval=0., maxval=2*jnp.pi)
        
        # Compute x and y coordinates of the sampled points on the circle
        x = params["Radius"] * jnp.cos(thetas)
        y = params["Radius"] * jnp.sin(thetas)
        
        event = jnp.column_stack([jnp.ones(N) / N, x, y])
        
        return event

    def initialize_circle(event, N, seed):
        return {"Radius" : R/2}

    def project_circle(params):

        radius = params["Radius"]
        params["Radius"] = jnp.maximum(radius, 0.)
        return params
    
    def hard_compute_circle(spectral_event):

        omegas = spectral_event[:,0]
        two_EE = spectral_event[:,1]
        E_tot = jnp.sum(two_EE)

        # Cumulative spectral function, inclusive and exclusive
        cumulative_spectral_function = jnp.cumsum(two_EE)
        cumulative_exclusive = cumulative_spectral_function - two_EE

        term1 = jnp.sin(jnp.pi / 2 / E_tot * (E_tot - cumulative_exclusive))
        term2 = jnp.sin(jnp.pi / 2 / E_tot * (E_tot - cumulative_spectral_function))
        
        r_opt =  2 * jnp.sum(omegas * (term1 - term2)) / jnp.pi
        semd_opt = jnp.sum(omegas**2 * two_EE) - 2 * r_opt**2

        return semd_opt, r_opt

    _springiness = Observable(sample_circle, name = "spRinginess", initializer=initialize_circle, projector=project_circle, hard_compute_function = hard_compute_circle)
    _springiness.compile()


    # %%%%%%%%%% DISK %%%%%%%%%%

    def sample_disk(params, N, seed):

        key = random.PRNGKey(seed)
        random_numbers = random.uniform(key, shape=(N,2), minval=0., maxval=1)
        rads = jnp.sqrt(random_numbers[:,0])
        thetas = 2 * jnp.pi * random_numbers[:,1]
        
        # Compute x and y coordinates of the sampled points on the circle
        x = params["Radius"] * rads * jnp.cos(thetas)
        y = params["Radius"] * rads * jnp.sin(thetas)
        
        event = jnp.column_stack([jnp.ones(N) / N, x, y])
        
        return event

    def initialize_disk(event, N, seed):
        return {"Radius" : R/2}

    def project_disk(params):

        radius = params["Radius"]
        params["Radius"] = jnp.maximum(radius, 0.)
        return params

    _sdiskiness = Observable(sample_disk, name = "spDiskiness", initializer=initialize_disk, projector=project_disk)
    _sdiskiness.compile()

    # %%%%%%%%%% n-sPRONGINESS %%%%%%%%%%

    def build_n_spronginess(n):

        def sample(params, N, seed):
            event = params["Points"]
            return event
        
        def initialize_n(events, N, seed):

            return {"Points" : kT_N(events, n, 0.5)}
        
        def project_n(params):

            event = params["Points"]

            temp = jnp.copy(event)
            zs = event[:,0]
            num_particles = event.shape[0]

            cnt_n = jnp.arange(num_particles)

            u = jnp.sort(zs, axis = -1)[::-1]
            v = (jnp.cumsum(u, axis = -1)-1) / (cnt_n + 1)
            w = v[jnp.sum(u > v, axis = -1) - 1]
            temp2 = temp.at[:,0].set(jnp.maximum(zs - w, 0))

            # temp = temp.at[:,:,0].set(jnp.maximum(temp[:,:,0], 0))
            # return temp

            return {"Points" : temp2}
        
        O = Observable(sample, name = f"{n}-sPronginess", initializer=initialize_n, projector=project_n)
        O.skip_vmap_initialization = True
        return O

    def _1spronginess_hard_compute_function(spectral_events):

        omegas = spectral_events[:,0]
        two_EE = spectral_events[:,1]
        E_tot = jnp.sum(two_EE)

        semd_opt = jnp.sum(omegas**2 * two_EE)

        return semd_opt, 0

    _1spronginess = build_n_spronginess(1)
    _1spronginess.hard_compute_function = _1spronginess_hard_compute_function
    _1spronginess.compile(N_sample = 1)

    _2spronginess = build_n_spronginess(2)
    _2spronginess.compile(N_sample = 2)

    _3spronginess = build_n_spronginess(3)
    _3spronginess.compile(N_sample = 3)

    observables_dict["spLineliness"] = _splineliness
    observables_dict["spRinginess"] = _springiness
    observables_dict["spDiskiness"] = _sdiskiness
    observables_dict["1-sPronginess"] = _1spronginess
    observables_dict["2-sPronginess"] = _2spronginess
    observables_dict["3-sPronginess"] = _3spronginess

    return observables_dict




# Get the spherical coordinates of the particles

def spherical_coordinates(data):
    p = data[:, 1:]
    r = jnp.linalg.norm(p, axis=-1)
    data[:,0] = r
    theta = jnp.arccos(p[:, 2] / r)
    phi = jnp.arctan2(p[:, 1], p[:, 0])

    spherical_data = jnp.stack([r, theta, phi], axis=-1)
    return spherical_data


def spherical_to_4vector(event):

    r = event[:,0]
    theta = event[:,1]
    phi = event[:,2]

    px = r * jnp.sin(theta) * jnp.cos(phi)
    py = r * jnp.sin(theta) * jnp.sin(phi)
    pz = r * jnp.cos(theta)

    return jnp.column_stack([r, px, py, pz])



def build_event_observables():

    observables_dict = {}

    # %%%%%%%%%% ISOTROPY %%%%%%%%%%

    def sample_uniform(params, N, seed):

        key = random.PRNGKey(seed)

        # Split the key
        key, subkey = random.split(key)

        phi = random.uniform(key, shape=(N,), minval=0., maxval=2*jnp.pi)
        costhtea = random.uniform(subkey, shape=(N,), minval=-1., maxval=1.)
        theta = jnp.arccos(costhtea)

        event = jnp.column_stack([jnp.ones(N) / N, theta, phi])
        event = spherical_to_4vector(event)

        return event
    

    def initialize_uniform(event, N, seed):
        return {}
    
    def project_uniform(params):
        return params




    _spisotropy = Observable(sample_uniform, name = "spIsotropy", initializer=initialize_uniform, projector=project_uniform, metric_type = "spherical", is_trivial=True)
    _spisotropy.compile()

    # %%%%%%%%%% RINGINESS %%%%%%%%%%

    def sample_ring(params, N, seed):

        key = key = random.PRNGKey(seed)

        phi = random.uniform(key, shape=(N,), minval=0., maxval=2*jnp.pi)
        theta = jnp.pi / 2 * jnp.ones_like(phi)

        event = jnp.column_stack([jnp.ones(N) / N, theta, phi])
        event = spherical_to_4vector(event)

        return event

    

    def initialize_ring(event, N, seed):
        return {}

    def project_ring(params):
        return params

    def hard_compute_ring(spectral_event):

        omegas = spectral_event[:,0]
        two_EE = spectral_event[:,1]
        E_tot = jnp.sum(two_EE)

        # Cumulative spectral function, inclusive and exclusive
        cumulative_spectral_function = jnp.cumsum(two_EE)
        cumulative_exclusive = cumulative_spectral_function - two_EE

        
        term1 = jnp.sum(two_EE * omegas * (omegas - 2*jnp.pi))
        term2 = jnp.pi**2 * E_tot**2 / 3
        term3 = (jnp.pi / E_tot**2) * jnp.sum(omegas * ((E_tot - cumulative_exclusive)**2 - (E_tot - cumulative_spectral_function)**2))

        semd_opt = term1 + term2 + term3
        
        return semd_opt, 0

    _springiness = Observable(sample_ring, name = "spRinginess", initializer=initialize_ring, projector=project_ring, hard_compute_function = hard_compute_ring, metric_type = "spherical", is_trivial=True)
    _springiness.compile()


    # %%%%%%%%%% PRONGINESS %%%%%%%%%%

    def build_n_pronginess(n):
            
            def sample(params, N, seed):
                event = params["Points"]
                return event
            
            def initialize_n(events, N, seed):
    
                return {"Points" : spherical_coordinates(kT_N(events, n, 0.5))}
            
            def project_n(params):
    
                event = params["Points"]
    
                temp = jnp.copy(event)
                zs = event[:,0]
                num_particles = event.shape[0]
    
                cnt_n = jnp.arange(num_particles)
    
                u = jnp.sort(zs, axis = -1)[::-1]
                v = (jnp.cumsum(u, axis = -1)-1) / (cnt_n + 1)
                w = v[jnp.sum(u > v, axis = -1) - 1]
                temp2 = temp.at[:,0].set(jnp.maximum(zs - w, 0))
    
                return {"Points" : temp2}
            
            O = Observable(sample, name = f"{n}-sPronginess", initializer=initialize_n, projector=project_n, metric_type = "spherical")
            O.skip_vmap_initialization = True
            return O
    
    
    _1spronginess = build_n_pronginess(1)
    _1spronginess.compile(N_sample = 1)

    _2spronginess = build_n_pronginess(2)
    _2spronginess.compile(N_sample = 2)

    _3spronginess = build_n_pronginess(3)
    _3spronginess.compile(N_sample = 3)

    # %%%%%%%%%% THRUST %%%%%%%%%%

    def sample_thrust(params, N, seed):

        key = random.PRNGKey(seed)

        phi = jnp.array([0., jnp.pi])
        theta = jnp.pi / 2 * jnp.ones_like(phi)
    
        event = jnp.column_stack([jnp.ones(2) / 2, theta, phi])
        event = spherical_to_4vector(event)

        return event
    
    def initialize_thrust(event, N, seed):
        return {}
    
    def project_thrust(params):
        return params

    
    
    _sthrust = Observable(sample_thrust, name = "sThrust", initializer=initialize_thrust, projector=project_thrust, metric_type = "spherical", is_trivial=True)
    _sthrust.compile()


    observables_dict["spIsotropy"] = _spisotropy
    observables_dict["spRinginess"] = _springiness
    observables_dict["1-sPronginess"] = _1spronginess
    observables_dict["2-sPronginess"] = _2spronginess
    observables_dict["3-sPronginess"] = _3spronginess
    observables_dict["sThrust"] = _sthrust





    return observables_dict
        
