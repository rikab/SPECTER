
import jax
from pyspecter.SpectralEMD_Helper import ds2_events1_spectral2



class Observable():

    def __init__(self, param_dict, sampler, name, initializer = None, projector = None):

        self.param_dict = param_dict
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

        self.is_trivial = (len(list(self.param_dict.keys())) == 0)


    def sample(self, n_samples, seed):

        return self.sampler(self.param_dict, n_samples, seed)
    

    def project(self, x):

        self.param_dict = self.projector(self.param_dict)


    def train_step(self, epoch, spectral_event, params, sample_N):

        shape_event = self.sample(params, sample_N, seed = epoch)
        sEMDS = jax.checkpoint(ds2_events1_spectral2)(shape_event, spectral_event)
        return sEMDS