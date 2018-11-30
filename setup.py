#!/usr/bin/env python
import setuptools
from numpy.distutils.core import setup, Extension
import os

ext = [Extension(name='Xana.XpcsAna.fecorrt3m',
                 sources=['Xana/XpcsAna/fecorrt3m.f90'],
                 f2py_options=['--verbose'])]

setup(ext_modules=ext)
