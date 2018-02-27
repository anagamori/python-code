#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:54:05 2018

@author: akiranagamori
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


ext_modules=[ Extension("TwitchBasedMuscleModel_v3",
              ["TwitchBasedMuscleModel_v3.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
    ext_modules = cythonize("TwitchBasedMuscleModel_v3.pyx")
) 