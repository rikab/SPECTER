import jax.numpy as jnp
from jax import random
from jax import grad, jacobian, jit


from pyspecter.Observables import Observable
from pyspecter.utils.data_utils import kT_N




def build_jet_observables(R = 0.5):


    observables_dict = {}

    def sample_line(params, N, seed):

        key = random.PRNGKey(seed)
        ones = random.uniform(key, shape=(N,), minval=0., maxval=1)
        
        # Compute x and y coordinates of the sampled points on the line
        x = params["Length"] * ones
        y = params["Length"] * jnp.zeros_like(ones)
        
        event = jnp.column_stack([jnp.ones(N) / N, x, y])
        
        return event 

    def initialize_line(event, N, seed):
        return {"Length" : R/2}

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
        semd_opt = jnp.sum(omegas**2 * two_EE) -   R**2 /6

        return semd_opt, l_opt





    _splineliness = Observable(sample_line, name = "spLineliness", initializer=initialize_line, projector=project_line, hard_compute_function = hard_compute_line)
    _splineliness.compile()

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

    _springiness = Observable(sample_circle, name = "spRinginess", initializer=initialize_circle, projector=project_circle)
    _springiness.compile()

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

    _1spronginess = build_n_spronginess(1)
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