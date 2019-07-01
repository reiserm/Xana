#!/usr/bin/env python
import setuptools
from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import os

ext = [Extension(name='Xana.XpcsAna.fecorrt3m',
                 sources=['Xana/XpcsAna/fecorrt3m.f90'],
                 f2py_options=['--verbose'])]

setup(
    name = 'Xana',
    version = '0.0.8',
    packages=setuptools.find_packages(),
    author = 'Mario Reiser',
    author_email = 'mario.mkel@gmail.com',
    url = 'https://git.xfel.eu/gitlab/reiserm/Xana.git',
    keywords = 'analysis XPCS XSVS XFEL X-ray',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Topic :: Data Analysis',
        'long_description = file: README.md'
    ],
    python_requires ='>= 3.6',
    install_requires = [
        'numpy',
        'pandas',
        'lmfit',
        'numpy',
        'pyfai',
        'cbf',
        'emcee',
        'corner',
        'h5py',
        'ipywidgets',
    ],
    ext_modules=ext,
)

