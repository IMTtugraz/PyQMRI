Installation Guide
========================

.. role:: bash(code)
   :language: bash
   
.. role:: python(code)
   :language: python
   
First make sure that you have a working OpenCL installation

* OpenCL is usually shipped with GPU driver (Nvidia/AMD)
* Install the ocl_icd and the OpenCL-Headers

  :bash:`apt-get install ocl_icd* opencl-headers`
    
Possible restart of system after installing new drivers and check if OpenCL is working

* Build clinfo_:
* Run clinfo_ in terminal and check for errors

Install clFFT library:  

* Either use the package repository,e.g.:

  :bash:`apt-get install libclfft*`

* Or download a prebuild binary of clfft_

  - Please refer to the clfft_ docs regarding building
  - If build from source symlink clfft_ libraries from lib64 to the lib folder and run :bash:`ldconfig`
    
Install gpyfft_ by following the instruction on the GitHub page. 
  
To Install PyQMRI, a simple
  
:bash:`pip install pyqmri`
    
should be sufficient to install the latest release.
    
Alternatively, clone the git repository and navigate to the root directory of PyQMRI. Typing
  
:bash:`pip install .`
    
should take care of the other dependencies using PyPI and install the package. 
     
In case OCL > 1.2 is present, e.g. by some CPU driver, and NVidia GPUs needs to be used the flag
PRETENED_OCL 1.2 has to be passed to PyOpenCL during the build process. This 
can be done by:

.. code-block:: bash

    ./configure.py --cl-pretend-version=1.2
    rm -Rf build
    python setup.py install
    
    
.. _clfft: https://github.com/clMathLibraries/clFFT
.. _gpyfft: https://github.com/geggo/gpyfft
.. _clinfo: https://github.com/Oblomov/clinfo
