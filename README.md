# SPECTER (I had to name the repo *something*)

Implementation of the Spectral EMD as outlined in https://arxiv.org/abs/2305.03751. 

## Some notes

* Putting a lowercase 's' in front of shape names to distinguish it from ordinary EMD (e.g. 2-sPronginess vs. 2-Pronginess)
* The spectral representation computation is extremely fast thanks to numpy tricks and can probably be made even faster with effort/cython



## Dependencies


Along with "standard python imports", We need the `SHAPER` package with extras (since we need the CMSOpenData Loader)

```
python -m pip install --upgrade 'pyshaper[all]'
```

But this isn't necessary if event data is provided some other way.

## Changelog

* 20 June 2023: coded spectral representation