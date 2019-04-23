# MBPQ
===================================

* Requires [PyOpenCL](https://github.com/inducer/pyopencl) package
* Requires [GPyFFT](https://github.com/geggo/gpyfft) package 
* Requires [bart](https://github.com/mrirecon/bart)

Currently runs only on GPUs due to a limitation in the GPyFFT package.

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
  - Build or download binarys of [clFFT](https://github.com/clMathLibraries/clFFT)
    - Please refer to the [clFFT](https://github.com/clMathLibraries/clFFT) docs regarding building
    - If build from source symlink clfft libraries from lib64 to the lib folder and run ``` ldconfig ```
  - Build [GPyFFT](https://github.com/geggo/gpyfft) 
    ```
    python setup.py build_ext bdist_wheel
    pip install ./dist/YOUR-WHEEL-NAME.whl
    ```
  - Navigate to the root directory of ISMRM_RRSG and typing
    ```
    pip install .
    ```
    should take care of the other dependencies using PyPI and install the package.
    
Please refer to the documentaiton of [bart](https://github.com/mrirecon/bart) for a detailed explanation on how to set up the toolbox.


Running the reconstruction:
-------------------------
First download the challenge data ([goettingen](http://wwwuser.gwdg.de/~muecker1/rrsg_challenge.zip), [NYU](https://cai2r.net/sites/default/files/software/rrsg_challenge.zip)), extract it, and copy the two .h5 files into the ISMRM_RRSG root folder.
Use any shell to navigate to the root folder of ISMRM_RRSG and simply type:
```
./run_acc
```
to run the reconstruction for brain and heart data with increasing accelaration.
After reconstruction is finished, the required plots will be automatically generated and saved in the root folder.

Regularization can be changed or turned off by changing the value of ```lambd``` in ```default.ini```. The .ini file will be automatically generated the first time the code is run. The ```tol``` parameter can be used to change the desired toleranze of the optimization scheme. ```max_iters``` defines the maximum number of CG iterations.

Citation:
----------
Please cite "Oliver Maier, Matthias Schloegl, Kristian Bredies, and Rudolf Stollberger; 3D Model-Based Parameter Quantification on Resource Constrained Hardware using Double-Buffering. Proceedings of the 27th meeting of the ISMRM, 2019, Montreal, Canada" if using the software or parts of it, specifically the PyOpenCL based NUFFT, in your work.
