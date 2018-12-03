# Xana - First Steps

## Install

You will need a Fortran compiler like `gfortran`.
During installation `f2py` (now part of `numpy`) is used to build python modules from fortran routines.

1. If a Fortran compiler is not already installed, install Gfortran:
   * Linux: `apt install gfortran`
   * Mac: `brew install gcc`
   * On Windows it is a little more complicated. A possible way
     to install a FORTRAN compiler is given [here](https://www.scivision.co/windows-gcc-gfortran-cmake-make-install/).
3. Download the Xana source code from gitlab
   ```sh
   git clone git clone https://git.xfel.eu/gitlab/reiserm/Xana.git
   ```
   
2. Install Xana code
   ```sh
   cd Xana
   pip install .
   ```
   
   
## Example Notebook
The folder `xpcs_example` contains an example notebook and an example
XPCS data set of 100nm SiO2 nanoparticles dispersed in glycerol. It
shows how to create a setup file and calculate correlation
functions. A detailed explanation is given in the notebook.
