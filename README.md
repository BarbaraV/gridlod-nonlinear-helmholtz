# gridlod-helmholtz-nonlinear

```
# This repository is part of the project for "Multiscale scattering in nonlinear Kerr-type media":
#   https://github.com/BarbaraV/gridlod-nonlinear-helmholtz
# Copyright holder: Roland Maier, Barbara Verfürth 
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```

This repository provides code for the experiments of the paper "Multiscale scattering in nonlinear Kerr-type media" by Roland Maier and Barbara Verfürth. The code is based on the module `gridlod`  which has been developed by Fredrik Hellman and Tim Keil and consists of code for PGLOD.  `gridlod` is  provided as a submodule, the other files in this repository extend and modify the PGLOD to work for nonlinear Helmholtz equation and were written by Roland Maier and Barbara Verfürth.

## Setup

This setup works with a Ubuntu system. The following packages are required (tested versions):
 - python (v3.8.5)
 - numpy (v1.19.4)
 - scipy (v1.5.4)
 - scikit-sparse
 - UMFPack 
 - matplotlib
 
Please see also the README of the `gridlod` submodule for required packages and setup.
Initialize the submodule via

```
git submodule update --init --recursive
```

Now, build and activate a python3 virtual environment with

```
virtualenv -p python3 venv3
. venv3/bin/activate
```

and execute the following command

```
echo $PWD/gridlod/ > $(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')/gridlod.pth
```
Now you can use every file from the submodule in the virtualenv. Install all the required python packages for gridlod. 

## Reproduction of experiments

To run the experiments that are presented in the paper, execute either

``` 
python run_exp1.py
``` 
for the experiments of Section 4.1 (produces Figures 4.1 and 4.2 and prints the mentioned percentages on screen)

or

``` 
python run_exp2.py
``` 
for the experiments of Section 4.2 (produces Figures 4.3 and 4.4 and prints the mentioned percentages on screen)

or

``` 
python run_exp3.py
``` 
for the experiment of Section 4.3 (produces Figures 4.5).
Note that this will take some time because the three methods are computed one after the other and the possible
parallelization of the code is not exploited. Alternatively, you can follow the instructions in the files and
run the code with the proposed adapted parameters for a qualitative illustration.

If you only want to see the error plots for the experiments in the paper, you just have to run

``` 
python plot.py
``` 
All data from the experiments are available and stored in the repository as .mat-files. 

## Note

This code is meant to illustrate the numerical methods presented in the paper. It is not optimized in any way, in particular no parallelization of the correcector problems is implemented. See `gridlod` for some further explanations of parallelization of the PGLOD with ipyparallel. In the sequential version provided here, the method may (and will) take longer to compute the solution, especially for the nonlinear problem, than a FEM on the fine reference mesh. This picture changes -- and, actually, these are the situations the method is targeted at -- if the same problem with many different right-hand sides should be solved or if a fine solution cannot be computed.
