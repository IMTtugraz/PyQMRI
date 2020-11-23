##################################
Welcome to PyQMRI's documentation!
##################################

3D model based parameter quantification for MRI.

PyQMRI is a Python module to quantify tissue parameters given a set of 
MRI measurements, specifically desinged to quantify the parameter of interest. 
Examples include T1 quantification from variable flip angle or 
inversion-recovery Look-Locker data, T2 quantification using a 
mono-exponential fit, or Diffusion Tensor quantification. 

In addition, a Genereal Model exists that can be invoced 
using a text file containing the analytical signal equation.

For a real world usage example have a look at the :doc:`Quickstart Guide <quickstart>`.
The example can also be run interactively using GoogleColab_.

.. toctree::
   :hidden:
   :includehidden:
   
   quickstart
   installation
   running
   fileformat
   configfile
   api
   genindex
   py-modindex
   searchindex

Sample Data
-----------
In-vivo datasets used in the original publication (doi: `[10.1002/mrm.27502]`_) can be found at zenodo_.
As these data sets are from an older release, the coil sensitivity profiles saved within the .h5 files
need to be deleted prior to reconstruction. This invokes a new conputation of coil sensitivity profiles,
matching the data within the fitting.


Limitations and known Issues:
------------------------------
Currently runs only on GPUs due to having only basic CPU support for the clfft_.

Citation:
----------
Please cite "Oliver Maier, Matthias Schloegl, Kristian Bredies, and Rudolf Stollberger; 3D Model-Based Parameter Quantification on Resource Constrained Hardware using Double-Buffering. Proceedings of the 27th meeting of the ISMRM, 2019, Montreal, Canada" if using the software or parts of it, specifically the PyOpenCL based NUFFT, in your work.

Older Releases:
----------------
You can find the code for 

| Maier O, Schoormans J,Schloegl M, Strijkers GJ, Lesch A, Benkert T, Block T, Coolen BF, Bredies K, Stollberger R 
| **Rapid T1 quantification from high resolution 3D data with model‐based reconstruction.**
| *Magn Reson Med.*, 2018; 00:1–16 doi: `[10.1002/mrm.27502]`_

at `[v0.1.0] <(https://github.com/IMTtugraz/PyQMRI/tree/v.0.1.0)>`_

.. _OpenCL: https://www.khronos.org/opencl/
.. _`[10.1002/mrm.27502]`: http://onlinelibrary.wiley.com/doi/10.1002/mrm.27502/full
.. _zenodo: https://doi.org/10.5281/zenodo.1410918
.. _clfft: https://github.com/clMathLibraries/clFFT
.. _GoogleColab: https://colab.research.google.com/drive/19BfSJmDPinZDY0m1sMAhETutIiJG3b33?usp=sharing
