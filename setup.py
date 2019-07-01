#!/usr/bin/env python
import setuptools
from setuptools.config import read_configuration
from numpy.distutils.core import setup, Extension
import os

conf_dict = read_configuration('setup.cfg')

ext = [Extension(name='Xana.XpcsAna.fecorrt3m',
                 sources=['Xana/XpcsAna/fecorrt3m.f90'],
                 f2py_options=['--verbose'])]

setup(ext_modules=ext, **conf_dict)
