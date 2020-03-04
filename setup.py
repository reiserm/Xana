#!/usr/bin/env python
import setuptools
from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import os
from distutils.command.sdist import sdist

# load the description from the README.md file
with open("README.md", "r") as fh:
    long_description = fh.read()

# setup f2py
ext = [Extension(name='Xana.XpcsAna.fecorrt3m',
                 sources=['Xana/XpcsAna/fecorrt3m.f90'],
                 f2py_options=['--verbose'])]

setup(
    cmdclass={'sdist': sdist},
    name = 'Xana',
    version = '0.0.8',
    packages=setuptools.find_packages(),
    license = 'MIT',
    author = 'Mario Reiser',
    author_email = 'mario.mkel@gmail.com',
    url = 'https://github.com/reiserm/Xana',
    download_url = 'https://github.com/reiserm/Xana/archive/v0.0.8-alpha.tar.gz',
    keywords = ['data analysis', 'XPCS', 'XSVS', 'SAXS',],
    description="Analysis software for XPCS, XSVS and SAXS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
  ],
    ext_modules=ext,
)

