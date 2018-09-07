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

A working python 3 installation with numpy, scipy, matplotlib, cython, pyfftw numexp, h5py, and ipyparallel. We highly recommend to use Anaconda.
To use Wavelet regularization the pywt package is required. 

The primaldual toolbox from [https://github.com/VLOGroup/imageutilities] build with gpuNUFFT

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
To display the results, load them using h5py and run the shipped multislice_viewer in a python session e.g. to diplay T1:
```
import h5py
import multislice_viewer as msv
import numpy as np

file = h5py.File("path_to_file")
some_results = file["tgv_full_result_0"][()]

msv.imshow(np.abs(some_result)[-1,1,....])
```
The first dimension contains the results after each GN step. Therefore, -1 is used to show the final results. The second dimension conotains proton density and T1 where proton density is located at 0 and T1 at position 1.
## Sample Data

In-vivo datasets used in the publication can be found at 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1410918.svg)](https://doi.org/10.5281/zenodo.1410918)

## License
This software is published under GNU GPLv3. In particular, all source code is provided "as is" without warranty of any kind, either expressed or implied. For details, see the attached LICENSE.
