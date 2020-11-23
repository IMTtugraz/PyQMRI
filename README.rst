.. |travis| image:: https://travis-ci.com/IMTtugraz/PyQMRI.svg?branch=master
    :target: https://travis-ci.com/IMTtugraz/PyQMRI
.. |gitlab| image:: https://gitlab.tugraz.at/F23B736137140D66/PyQMRI/badges/master/pipeline.svg
   :target: https://gitlab.tugraz.at/F23B736137140D66/PyQMRI/-/commits/master
.. |pypi| image:: https://badge.fury.io/py/pyqmri.svg
    :target: https://pypi.org/project/pyqmri
.. |docs| image:: https://readthedocs.org/projects/pyqmri/badge/?version=latest
    :target: https://pyqmri.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/19BfSJmDPinZDY0m1sMAhETutIiJG3b33?usp=sharing
    :alt: Open in Colab  
    
Travis: |travis| GitLab: |gitlab| |pypi| |docs|
    
.. role:: bash(code)
   :language: bash
.. role:: python(code)
   :language: python
   
PyQMRI - Model-Based Parameter Quantification
=============================================
PyQMRI is a Python based toolbox for quantitative Magnetic Resonance Imaging (MRI). Utilizing _PyOpenCL and a double-buffering scheme, 
it enables the accelerated reconsruction and fitting of arbitrary large datasets on memory limited GPUs.
Currently, PyQMRI supports the processing of 3D Cartesian and non-Cartesian (stack-of-) data.

Examples include T1 quantification from variable flip angle or 
inversion-recovery Look-Locker data, T2 quantification using a 
mono-exponential fit, or Diffusion Tensor quantification. 

In addition, a Genereal Model exists that can be invoced 
using a text file containing the analytical signal equation.

For a real world usage example have a look at the `Quickstart Guide`_.
The example can also be run interactively using |Colab|.

Installation and usage guides, as well as API documentaiton, can be found in the Documentation_


Sample Data
-----------
In-vivo datasets used in the original publication (doi: `[10.1002/mrm.27502]`_) can be found at zenodo_. If you use the sample data with the recent release of PyQMRI please delete the "Coils"
entry in the .h5 to force a recomputation of the receive coil sensitivities as the orientation does not match the data.


Contributing
------------
If you want to contribute please make sure that all tests pass and adhere to our `Code of Conduct`_. 
Prior to running the tests it is necessary to start an ipcluster. 
An exemplary workflow would be:
:bash:`ipcluster start &`
followed by typing
:bash:`pytest test`
in the PyQMRI root folder. It is advised to run unit and integration tests after each other as OUT_OF_MEMORY exceptions can occur if both are in one session, e.g.:
:bash:`pytest test/unittests`
:bash:`pytest test/integrationtests`

For more detailed instructions on how to contribute have a look at contributing_.


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

at `[v0.1.0] <https://github.com/IMTtugraz/PyQMRI/tree/v.0.1.0>`_

.. _OpenCL: https://www.khronos.org/opencl/
.. _clfft: https://github.com/clMathLibraries/clFFT
.. _gpyfft: https://github.com/geggo/gpyfft
.. _clinfo: https://github.com/Oblomov/clinfo
.. _`[10.1002/mrm.27502]`: http://onlinelibrary.wiley.com/doi/10.1002/mrm.27502/full
.. _zenodo: https://doi.org/10.5281/zenodo.1410918
.. _NLINV: https://doi.org/10.1002/mrm.21691
.. _PyOpenCL: https://github.com/inducer/pyopencl
.. _GoogleColab: https://colab.research.google.com/drive/19BfSJmDPinZDY0m1sMAhETutIiJG3b33?usp=sharing
.. _contributing: CONTRIBUTING.rst
.. _`Quickstart Guide` : https://pyqmri.readthedocs.io/en/latest/quickstart.html
.. _Documentation : https://pyqmri.readthedocs.io/en/latest/?badge=latest
.. _`Code of Conduct` : CODE_OF_CONDUCT.rst
