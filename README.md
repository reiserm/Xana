[![Tests](https://github.com/reiserm/Xana/workflows/Tests/badge.svg)](https://github.com/reiserm/Xana//actions?query=workflow%3ATests)
[![Codecov](https://codecov.io/gh/reiserm/Xana/branch/master/graph/badge.svg)](https://codecov.io/gh/reiserm/Xana)

# Documentation

The documentation can be found on Read the Docs under this [link](https://xana.readthedocs.io/en/latest/index.html).

# Install Xana

A Fortran compiler and `f2py` (now part of `numpy`) are required build python
modules from fortran routines during installation. If the installation fails due
to a missing compiler see point 3.

1. Install Xana using pip:
   ```sh
   pip install Xana
   ```

2. Install most recent Xana version from GitHub:
   Download (clone) the repository and install Xana.
   ```sh
   git clone https://github.com/reiserm/Xana.git
   cd Xana
   pip install .
   ```
   or use `pip install -e .` for editable installation. Then you can update Xana
   by executing `git pull` from within the Xana directory.

   Install the latest version from GitHub directly with pip:
   ```sh
   pip install git+https://github.com/reiserm/Xana.git
   ```


# Example Data

An example XPCS dataset can be downloaded from
[Xana_example](https://github.com/reiserm/Xana_example). The repository contains
* Example XPCS data measured with 100nm (diameter) SiO2 nanoparticles dispersed
in a glycerol-water mixture.
* A mask of the detector, i.e., a 2D array where bad or broken pixels are 0 and
  others are 1.
* A tutorial Jupyter notebook.

Use
```sh
git clone https://github.com/reiserm/Xana_example.git
tar xzf ./Xana_example/xpcs_data.tgz
```
to download and unpack the data.
