#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:17:32 2018

@author: akiranagamori
"""


from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


ext_modules=[ Extension("fastloop",
              ["rear.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
    ext_modules = cythonize("test.pyx")
) 