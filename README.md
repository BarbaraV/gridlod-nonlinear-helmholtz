# gridlod-helmholtz-nonlinear

```
# This repository is part of the project for "Multiscale scattering in nonlinear Kerr-type media":
#   https://
# Copyright holder: Roland Maier, Barbara Verfürth 
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```

This repository provides code for the experiments of the paper "Multiscale scattering in nonlinear Kerr-type media" by Roland Maier and Barbara Verfürth. The code is based on the module `gridlod`  which has been developed by Fredrik Hellman and Tim Keil and consists of code for PGLOD.  `gridlod` is  `provided as a submodule, the other files in this repository extend and modify the PGLOD to work for nonlinear Helmholtz equation and were written by Roland Maier and Barbara Verfürth.

## Setup

This setup works with a Ubuntu system. The following packages are required (tested versions):
 - python (v3.)
 - numpy
 - scipy
 - UMFPack 
 - matplotlib
Please see also the README of the `gridlod` submodule for required packages and setup.
Initialize the submodule via

```
git submodule update --init --recursive
```

and execute the following commands 

```
echo $PWD/gridlod/ > $(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')/gridlod.pth
```
Now you can use every file from the submodule. Install all the required python packages for gridlod.

## Reproduction of experiments

To run the experiments that are presented in the paper, execute either

``` 
python experiments/
``` 
for the experiments of Section 4.1 (produces Figures .... and prints the mentioned percentages on screen)

or

``` 
python experiments/
``` 
for the experiments of Section 4.1 (produces Figures .... and prints the mentioned percentages on screen)

If you only want to see the results, you just have to run

``` 
python experiments/plot_xy.py
``` 
with xy = for Figure ..., xy = for Figure ... All data from the experiments are available and stored in the repository. 

## Note

This code is meant to illustrate the numerical methods presented in the paper. It is not optimized in any way, in particular no parallelization of the correcector problems is implemented. See `gridlod` for some further explanations of parallelization of the PGLOD with ipyparallel. In the sequential version provided here, the method may (and will) take longer to compute the solution, especially for the nonlinear problem, than a FEM on the fine reference mesh. This picture changes -- and, actually, these are the situations the method is targeted at -- if the same problem with many different right-hand sides should be solved or if a fine solution cannot be computed.
