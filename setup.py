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
    license = 'MIT',
    author = 'Mario Reiser',
    author_email = 'mario.mkel@gmail.com',
    url = 'https://github.com/reiserm/Xana',
    keywords = ['data analysis', 'XPCS', 'XSVS', 'SAXS',],
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
    classifiers=[
        'Development Status :: 3 - Alpha',      
        'Intended Audience :: Scientists',    
        'Topic :: Data Analysis :: Coherent X-ray Scattering',
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3',      
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'long_description = file: README.md'
  ],
    ext_modules=ext,
  
)

