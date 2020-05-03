#!/usr/bin/env python


import os
import sys
import setuptools
from setuptools import find_packages
from distutils.command.sdist import sdist


def install_package():

    # load the description from the README.md file
    with open("README.md", "r") as fh:
        long_description = fh.read()


    cmdclass = {'sdist': sdist}
    metadata = dict(
        cmdclass = cmdclass,
        name = 'Xana',
        version = '0.0.10',
        packages = setuptools.find_packages(),
        license = 'MIT',
        author = 'Mario Reiser',
        author_email = 'mario.mkel@gmail.com',
        url = 'https://github.com/reiserm/Xana',
        download_url = "https://pypi.python.org/pypi/Xana",
        keywords = ['data analysis', 'XPCS', 'XSVS', 'SAXS',],
        description="Analysis software for XPCS, XSVS and SAXS data.",
        long_description = long_description,
        long_description_content_type="text/markdown",
        python_requires ='>= 3.6',
        install_requires = [
            'numpy',
            'pandas',
            'lmfit',
            'pyfai',
            'cbf',
            'matplotlib',
            'emcee',
            'corner',
            'h5py',
            'seaborn',
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
    )

    args = sys.argv[1:]

    build_ext = False
    if 'build_ext' in args:
        build_ext = True

    if build_ext:

        try:
            from numpy.distutils.core import setup, Extension
            # setup f2py
            ext = [Extension(name='Xana.XpcsAna.fecorrt3m',
                        sources=['Xana/XpcsAna/fecorrt3m.f90'],
                        f2py_options=['--verbose'])]

            metadata['ext_modules'] = ext
        except ModuleNotFoundError:
            print('Failed building fortran extensions.')
            from setuptools import setup

    else:
        from setuptools import setup

    setup(**metadata)


if __name__ == "__main__":
    install_package()
