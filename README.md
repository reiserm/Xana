# Xana - README

## Install

You will need a Fortran compiler like `gfortran`. During installation `f2py`
(now part of `numpy`) is used to build python modules from fortran routines.

1. If a Fortran compiler is not already installed, install Gfortran:
   * Linux: `apt install gfortran`
   * Mac: `brew install gcc`
   * On Windows it is a little more complicated. A possible way to install a
     FORTRAN compiler is given
     [here](https://www.scivision.co/windows-gcc-gfortran-cmake-make-install/)
     or
     [here](https://www.scivision.dev/f2py-running-fortran-code-in-python-on-windows/).
     
2. Download the Xana source code from gitlab
   ```sh
   git clone https://git.xfel.eu/gitlab/reiserm/Xana.git
   ```
3. On OSX: if building the module fails, it might help to set the following environment variables:
   ```sh
   unset LDFLAGS
   unset SHELL
   ```
   
4. Install Xana
   ```sh
   cd Xana
   pip install .
   ```
   
   
## Example Notebook

The Xana example has been moved to a new repository:
[Xana_example](https://github.com/reiserm/Xana_example). It contains an example
notebook and an example XPCS data set of 100nm SiO2 nanoparticles dispersed in
glycerol. It shows how to create a setup file and calculate correlation
functions. A detailed explanation is given in the notebook.
