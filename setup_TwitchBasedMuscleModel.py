#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 22:40:43 2018

@author: akiranagamori
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


ext_modules=[ Extension("TwitchBasedMuscleModel",
              ["TwitchBasedMuscleModel.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
    ext_modules = cythonize("TwitchBasedMuscleModel.pyx")
) 