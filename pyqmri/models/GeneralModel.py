#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the general model for fitting."""
import configparser
import numpy as np
import sympy
from pyqmri.models.template import BaseModel, constraints


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ValueError('Boolean value expected.')


class Model(BaseModel):
    """Realization of a generative model based on sympy.

      This model can handel all kinds of sympy input in form of a config file.
      Partial derivatives of the model are automatically generated and a
      numpy compatible function is build from sumpy equations.

    Attributes
    ----------
      signaleq : sympy derived function
        The signal equation derived from sympy
      grad : list of functions
        Partial derivatives with respect to the unknowns
      rescalefun : list of functions
      Functions to rescale each parameter
      modelparams : list
        List of model parameters
      indphase : bool
        Flag to estimate the phase from a given image series.
        The phase is normed on the first image.
        If True, each image will be multiplied by the estimated phase in the
        forward and gradient evaluation.
      init_values : list of str
          Initial guess for each unknown
    """

    def __init__(self, par):

        super().__init__(par)
        config = configparser.ConfigParser()

        if not par["modelfile"].endswith('.ini'):
            par["modelfile"] += '.ini'
        try:
            with open(par["modelfile"], 'r') as f:
                config.read_file(f)
        except BaseException:
            print("Model file not readable or not found")
            raise BaseException
        finally:
            params = {}
            for key in config[par["modelname"]]:
                params[key] = config[par["modelname"]][key]

        modelpar = sympy.symbols(params["parameter"])
        unknowns = sympy.symbols(params["unknowns"])
        self._unknowns = unknowns

        par["unknowns_TGV"] = len(unknowns)
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        uk_scale = []
        for j in range(par["unknowns"]):
            uk_scale.append(sympy.symbols(str(unknowns[j])+"_sc"))
            params["signal"] = params["signal"].replace(
                str(unknowns[j]),
                "("+str(unknowns[j])+"*"+str(unknowns[j])+"_sc)")
            params["rescale"] = params["rescale"].replace(
                str(unknowns[j]),
                "("+str(unknowns[j])+"*"+str(unknowns[j])+"_sc)")
        signaleq = sympy.sympify(params["signal"])
        params["rescale"] = params["rescale"].split(",")

        self.grad = []
        self.rescalefun = []

        for uk, scalefuns in zip(unknowns, params["rescale"]):
            tmp_grad = sympy.diff(signaleq, uk)
            self.grad.append(sympy.lambdify(
                (modelpar, unknowns, uk_scale), tmp_grad))
            self.rescalefun.append(
                sympy.lambdify(
                    (modelpar, unknowns, uk_scale),
                    sympy.sympify(scalefuns)))

        self.signaleq = sympy.lambdify(
            (modelpar, unknowns, uk_scale), signaleq)

        self.modelparams = []
        for mypar in modelpar:
            tmp = par[str(mypar)]
            if np.isscalar(tmp) or tmp.shape == (1,):
                self.modelparams.append(tmp)
            elif (len(tmp.shape) >= 3 and
                  tmp.shape[-3:] == (self.NSlice, self.dimY, self.dimX)):
                self.modelparams.append(tmp)
            else:
                while len(tmp.shape) < 4:
                    tmp = tmp[..., None]
                self.modelparams.append(tmp)

        self.uk_scale = []
        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        box_constraints_low = params["box_constraints_lower"].split(",")
        box_constraints_up = params["box_constraints_upper"].split(",")
        real_const = params["real_value_constraints"].split(",")
        for uk in range(par["unknowns"]):
            self.constraints.append(
                constraints(
                    float(box_constraints_low[uk]),
                    float(box_constraints_up[uk]),
                    _str2bool(real_const[uk])))

        self.indphase = _str2bool(params["estimate_individual_phase"])

        self.init_values = params["guess"].split(",")

        self._plot = []
        self._phase = None
        self.guess = None

    def rescale(self, x):
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling factor and
        an optional transformation.

        Parameters
        ----------
          x : numpy.array
            The array of unknowns to be rescaled

        Returns
        -------
          numpy.array:
            The rescaled unknowns
        """
        tmp_x = np.copy(x)
        uk_name = []
        for j in range(x.shape[0]):
            tmp_x[j] = self.rescalefun[j](
                self.modelparams, x, self.uk_scale)
            uk_name.append(str(self._unknowns[j]))
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return {"data": tmp_x,
                "unknown_name": uk_name,
                "real_valued": const}

    def _execute_forward_3D(self, x):
        S = self.signaleq(self.modelparams, x, self.uk_scale)
        while len(S.shape) >= 5:
            S = np.squeeze(S, axis=0)
        if self.indphase is True:
            S *= self._phase
        S[~np.isfinite(S)] = 1e-20
        S = S.astype(dtype=self._DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        modelgradient = []
        if self.indphase is True:
            for ukgrad in self.grad:
                modelgradient.append(
                    ukgrad(self.modelparams, x, self.uk_scale)*self._phase)
        else:
            for ukgrad in self.grad:
                modelgradient.append(
                    ukgrad(self.modelparams, x, self.uk_scale))
        modelgradient = np.array(modelgradient, dtype=self._DTYPE)
        while len(modelgradient.shape) >= 6:
            modelgradient = np.squeeze(modelgradient, axis=1)
        modelgradient[~np.isfinite(modelgradient)] = 1e-20
        return modelgradient

    def computeInitialGuess(self, *args):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting, based
        on the values on the text file.

        Parameters
        ----------
          args : list of objects
            Assumes the image series at potition 0 and optionally computes
            a phase based on the difference between each image series minus
            the first image in the series. (Scan i minus Scan 0)
        """
        if self.indphase is True:
            self._phase = np.exp(1j*(np.angle(args[0])-np.angle(args[0][0])))
        x = np.ones((len(self.init_values),
                     self.NSlice, self.dimY, self.dimX), self._DTYPE)
        for j in range(len(self.init_values)):
            if "image" in self.init_values[j]:
                x[j] = args[0][int(self.init_values[j].split("_")[-1])]
            else:
                x[j] *= float(self.init_values[j])
        self.guess = x


def genDefaultModelfile():
    """Generate a default model config file.

    This method generates a default model file in the current project folder.
    This file can be modified or further models can be added.
    """
    config = configparser.ConfigParser()

    config["MonoExp"] = {}
    config["MonoExp"]["parameter"] = "TE"
    config["MonoExp"]["unknowns"] = "M0 A1"
    config["MonoExp"]["signal"] = "M0*exp(-TE*A1)"
    config["MonoExp"]["box_constraints_lower"] = "0,0"
    config["MonoExp"]["box_constraints_upper"] = "100,1"
    config["MonoExp"]["real_value_constraints"] = "False,True"
    config["MonoExp"]["guess"] = "1,0.01"
    config["MonoExp"]["rescale"] = "M0,1/A1"
    config["MonoExp"]["estimate_individual_phase"] = "False"

    config["VFA-E1"] = {}
    config["VFA-E1"]["parameter"] = "TR fa fa_corr"
    config["VFA-E1"]["unknowns"] = "M0 E_1"
    config["VFA-E1"]["signal"] = "M0*sin(fa*fa_corr)*"\
                                 "(1-E_1)/(1-E_1*cos(fa*fa_corr))"
    config["VFA-E1"]["box_constraints_lower"] = "0,0.9048"
    config["VFA-E1"]["box_constraints_upper"] = "10,0.99909"
    config["VFA-E1"]["real_value_constraints"] = "False,True"
    config["VFA-E1"]["guess"] = "1,0.99667"
    config["VFA-E1"]["rescale"] = "M0,-TR/log(E_1)"
    config["VFA-E1"]["estimate_individual_phase"] = "False"

    with open('models.ini', 'w') as configfile:
        config.write(configfile)
