"""
3D model based parameter quantification for MRI.

PyQMRI is a Python module to quantify tissue parameters given a set of MRI
measurements, specifically desinged to quantify the parameter of interest.
Examples include T1 quantification from variable flip angle or
inversion-recovery Look-Locker data, T2 quantification using a
mono-exponential fit, or Diffusion Tensor quantification. In addition,
a Genereal Model exists that can be invoked using a text file containing
the analytical signal equation.

See https://pyqmri.readthedocs.io/en/latest/ for a complete documentation.

"""
from pyqmri import operator
from pyqmri import streaming
from pyqmri import solver
from pyqmri import transforms
from pyqmri import models
from pyqmri.pyqmri import run
from pyqmri._helper_fun.multislice_viewer import imshow as msv
from pyqmri.models.template import BaseModel, constraints
from pyqmri.models.GeneralModel \
    import genDefaultModelfile as generate_text_models
