# MBPQ
===================================
* Requires [OpenCL](https://www.khronos.org/opencl/) >= 1.2
* Requires [clfft](https://github.com/clMathLibraries/clFFT)
* Requires Cython, pip >= 19, python >= 3.6

The Software is tested on Linux using the latest Nvidia driver but should be compatible with older drivers as well as different hardware (AMD). The following Installation Guide is targeted at Ubuntu but should work on any distribution provided the required packages are present (could be differently named).

* It is highly recommended to use an Anaconda environment

Quick Installing Guide:
---------------
First make sure that you have a working OpenCL installation
  - OpenCL is usually shipped with GPU driver (Nvidia/AMD)
  - Install the ocl_icd and the OpenCL-Headers
    ```
    apt-get install ocl_icd* opencl-headers
    ```  
Possible restart of system after installing new drivers
  - Build [clinfo](https://github.com/Oblomov/clinfo)
  - Run clinfo in terminal and check for errors
  
Install clFFT library:  
  - Either use the package repository,e.g.:
    ```
    apt-get install libclfft*
    ```  
  - Or download a prebuild binary of [clfft](https://github.com/clMathLibraries/clFFT) 
    - Please refer to the [clFFT](https://github.com/clMathLibraries/clFFT) docs regarding building
    - If build from source symlink clfft libraries from lib64 to the lib folder and run ``` ldconfig ```
    
  - Navigate to the root directory of MBPQ and typing
    ```
    pip install .
    ```
    should take care of the other dependencies using PyPI and install the package.
    
## Sample Data

In-vivo datasets used in the original publication (doi: [10.1002/mrm.27502](http://onlinelibrary.wiley.com/doi/10.1002/mrm.27502/full)) can be found at 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1410918.svg)](https://doi.org/10.5281/zenodo.1410918)    

Prerequests on the .h5 file:
-------------------------
The toolbox expects a .h5 file with a certain structure. 
  - kspace data (assumed to be 5D for VFA) and passed as:
    - real_dat (Scans, Coils, Slices, Projections, Samples)
    - imag_dat (Scans, Coils, Slices, Projections, Samples)
    
    If radial sampling is used the trajectory is expected to be:
    - real_traj (Scans, Projections, Samples)
    - imag_traj (Scans, Projections, Samples)
    
    Density compensation is performed internally assuming a simple ramp
    
    For Cartesian data Projections and Samples are replaced by ky and kx encodings points and no trajectory is needed.  
    
    
  - flip angle correction (optional) can be passed as:
    - fa_corr (Scans, Coils, Slices, dimY, dimX)
  - The image dimension for the full dataset is passed as attribute consiting of:
    - image_dimensions = (dimX, dimY, NSlice)
  - Parameters specific to the used model (e.g. TR or flip angle) need to be set as attributes e.g.:
    - TR = 5.38
    - flip_angle(s) = (1,3,5,7,9,11,13,15,17,19)
    
    The specific structure is determined according to the Model file.
    
  If predetermined coil sensitivity maps are available they can be passed as complex dataset, which can directly saved using Python. Matlab users would need to write/use low level hdf5 functions to save a complex array to .h5 file. Coil sensitivities are assumed to have the same number of slices as the original volume and are intesity normalized. The corresponding .h5 entry is named "Coils". If no "Coils" parameter is found the coil sensitivites are determined using the [NLINV](https://doi.org/10.1002/mrm.21691) algorithm and saved into the file. Additionally, if the number of slices is less than the number of reconstructed slices, sensitivities will get estimated.
    
Running the reconstruction:
-------------------------    
First, start an ipyparallel cluster on your local machine to speed up coil sensitivity estimation:
```
ipcluster start &
```
Reconstruction of the parameter maps can be started either using the terminal by typing:
```
mbpq
```
or from python by:
```
import mbpq
mbpq.mbpq()
```
A list of accepted flags can be printed using 
```
mbpq -h
```
or by fewing the documentation in of mbpq.mbpq in python.

Limitations and known Issues:
-------------------------
Currently runs only on GPUs due to having only basic CPU support for the clfft.

Citation:
----------
Please cite "Oliver Maier, Matthias Schloegl, Kristian Bredies, and Rudolf Stollberger; 3D Model-Based Parameter Quantification on Resource Constrained Hardware using Double-Buffering. Proceedings of the 27th meeting of the ISMRM, 2019, Montreal, Canada" if using the software or parts of it, specifically the PyOpenCL based NUFFT, in your work.
