from os.path import abspath
import logging
import numpy
from distutils.command.sdist import sdist

import setuptools
from distutils.core import setup
from Cython.Build import cythonize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Xana.setup")

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules = cythonize(abspath("Xana/XpcsAna/cpy_ecorr.pyx"), annotate=True)

setup(
    name="Xana",
    version="0.0.12",
    packages=setuptools.find_packages(),
    license="MIT",
    author="Mario Reiser",
    author_email="mario.mkel@gmail.com",
    url="https://github.com/reiserm/Xana",
    download_url="https://github.com/reiserm/Xana/archive/v0.0.12-alpha.tar.gz",
    keywords=[
        "data analysis",
        "XPCS",
        "XSVS",
        "SAXS",
    ],
    description="Analysis software for XPCS, XSVS and SAXS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">= 3.6",
    cmdclass={"sdist": sdist},
    install_requires=[
        "cbf>=1.0",
        "corner>=2.0",
        "emcee>=3.0",
        "h5py>=2.0",
        "hdf5plugin",
        "ipywidgets",
        "lmfit>=1.0",
        "matplotlib",
        "numpy>=1.19",
        "pandas>=1.0",
        "pyfai>=0.19",
        "seaborn",
        "cython",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "nbval",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
