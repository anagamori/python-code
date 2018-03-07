#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:00:02 2018

@author: akiranagamori
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


ext_modules=[ Extension("TwitchBasedMuscleModel_Synergists",
              ["TwitchBasedMuscleModel_Synergists.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
    ext_modules = cythonize("TwitchBasedMuscleModel_Synergists.pyx")
) 