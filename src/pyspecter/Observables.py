

class Observable():

    def __init__(self, param_dict, sampler, initializer = None):

        self.param_dict = param_dict
        self.sampler = sampler
        self.initializer = initializer

        self.is_trivial = (len(list(self.param_dict.keys())) == 0)


    def sample(self, n_samples):

        return self.sampler(n_samples, **self.param_dict)
    

    def enforce(self):

        pass
