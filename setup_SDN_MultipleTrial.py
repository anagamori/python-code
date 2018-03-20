#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:25:34 2018

@author: akira
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[ Extension("SDN_MultipleTrial",
              ["SDN_MultipleTrial.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math","-fopenmp"],
              extra_link_args = ['-fopenmp'])
]

setup(
    cmdclass = {"build_ext":build_ext},
    ext_modules = cythonize("SDN_MultipleTrial.pyx")
) 