# SPECTER 

Implementation of the Spectral EMD as outlined in https://arxiv.org/abs/2305.03751. 

## Usage

See `examples/pairwise_spectral_emd.ipynb` for an end-to-end example of how to use SPECTER.

## Installation

To install SPECTER, first clone this repository. Then, in the root directory of the repository, run the following command:

```bash
pip install -.
```



## Dependencies

The primary dependencies are `jax` and `jaxlib`. 

To install jax and jaxlib, run the following commands:

```bash
pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Changelog

* 20 July 2023: JAX-enabled gradient optimization
* 20 June 2023: coded spectral representation

## Contact

Please contact me at rikab@mit.edu with any questions, concerns, comments, or found bugs.