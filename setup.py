import logging
from distutils.command.sdist import sdist

import setuptools
from numpy.distutils.core import Extension, setup


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Xana.setup")

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules = [
    Extension(
        name='Xana.XpcsAna.fecorrt3m',
        sources=['Xana/XpcsAna/fecorrt3m.f90'],
        f2py_options=['--verbose'],
    )
]

setup(
    name='Xana',
    version='0.0.12',
    packages=setuptools.find_packages(),
    license='MIT',
    author='Mario Reiser',
    author_email='mario.mkel@gmail.com',
    url='https://github.com/reiserm/Xana',
    download_url='https://github.com/reiserm/Xana/archive/v0.0.12-alpha.tar.gz',
    keywords=['data analysis', 'XPCS', 'XSVS', 'SAXS', ],
    description='Analysis software for XPCS, XSVS and SAXS data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>= 3.6',
    ext_modules=ext_modules,
    cmdclass={'sdist': sdist},
    install_requires=[
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
