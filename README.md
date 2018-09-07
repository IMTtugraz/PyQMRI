# MBPQ

A framework for T1 quantification from variable flip angle and inversion recovery Look-Locker data

If you use this software please cite:
* Maier O, Schoormans J,Schloegl M, Strijkers GJ, Lesch A, Benkert T, Block T, Coolen BF, Bredies K, Stollberger R <br>
  __Rapid T1 quantification from high
resolution 3D data with model‐based reconstruction.__<br>
  _Magn Reson Med._, 2018; 00:1–16<br>
  doi: [10.1002/mrm.27502](http://onlinelibrary.wiley.com/doi/10.1002/mrm.27502/full)

## Getting Started

### Prerequisites

A working python 3 installation with cython, pyfftw and ipyparallel
Imageutilities from [https://github.com/VLOGroup/imageutilities] build with gpuNUFFT

### Installing

Prior to any use run
```
python setup.py build_ext --inplace
```
in the root folder of the project.

## How to
First start a ipyparallel cluster in the project folder:
```
ipcluster start &
```
To run the VFA reconstruction simple type:
```
python VFA_Init.py 
```
If no further options are specified this runs 3D reconstruction with TGV, TV and Wavelet constrained after each other.
To see the available options type:
```
python VFA_Init.py -h
```
IRLL reconstruction can be started with:
```
python IRLL_Init.py 
```

## Sample Data

In-vivo datasets used in the publication can be found at 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1410918.svg)](https://doi.org/10.5281/zenodo.1410918)

## License
This software is published under GNU GPLv3. In particular, all source code is provided "as is" without warranty of any kind, either expressed or implied. For details, see the attached LICENSE.
