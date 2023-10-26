from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

extensions = [
    Extension("find_pairs", sources=["find_pairs.pyx"], extra_compile_args=["-O3"], include_dirs=[np.get_include()])
]

setup(
    ext_modules = cythonize("find_pairs.pyx")
)
