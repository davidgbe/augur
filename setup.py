from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(Extension(
           "firegod",                                # the extesion name
           sources=["firegod.pyx", "perceptron.cpp"], # the Cython source and
                                                  # additional C++ source files
           language="c++",                        # generate and compile C++ code
           include_dirs=[numpy.get_include()]
      )))
