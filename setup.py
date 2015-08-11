from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(
    Extension(
        "augur",
        sources=["augur.pyx", "neural_net.cpp", "layer.cpp", "perceptron.cpp"],
        language="c++",    
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"]
    )
))
