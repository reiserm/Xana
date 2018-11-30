# Xana --- First Steps

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
