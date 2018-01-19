#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:22:54 2018

@author: akiranagamori
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("hello_world.pyx")
)