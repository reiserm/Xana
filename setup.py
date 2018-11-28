#!/usr/bin/env python
import setuptools
from numpy.distutils.core import setup, Extension
import os

ext = [Extension(name='fecorrt3m',
                 sources=['Xana/XpcsAna/fecorrt3m.f90'],
                 f2py_options=['--quiet'])]

setup(ext_modules=ext)
