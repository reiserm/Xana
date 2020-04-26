# Readme

## Installation

You need a Fortran compiler as `f2py` (now part of `numpy`) is used during
installation to build python modules from fortran routines. If the installation
fails due to a missing compiler see point 3.

1. Install Xana using pip:
    ```sh
    pip install Xana
    ```

2. Install Xana from github:
   ```sh
   git clone https://github.com/reiserm/Xana.git
   cd Xana
   pip install .
   ```

3. If a Fortran compiler is not already installed, try
   * Linux: `apt install gfortran`
   * MacOSX: `brew install gcc`
   * On Windows it is a little more complicated. A possible way to install a
     FORTRAN compiler is shown
     [here](https://www.scivision.co/windows-gcc-gfortran-cmake-make-install/)
     or
     [here](https://www.scivision.dev/f2py-running-fortran-code-in-python-on-windows/).
     
4. Additional remarks on the installation on OSX: if building the module fails, it might help to set the following environment variables:
   ```sh
   unset LDFLAGS
   unset SHELL
   ```
   
   
   
## Example Notebook

The Xana example has been moved to a new repository:
[Xana_example](https://github.com/reiserm/Xana_example). It contains an example
notebook and an example XPCS data set of 100nm SiO2 nanoparticles dispersed in
glycerol. It shows how to create a setup file and calculate correlation
functions. A detailed explanation is given in the notebook.
