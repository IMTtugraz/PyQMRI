#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the bi-exponential model for fitting."""
import numpy as np
from pyqmri.models.template import BaseModel, constraints
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Model(BaseModel):
    """Bi-exponential model for MRI parameter quantification.

    This class holds a bi-exponential model for fitting complex MRI data.
    It realizes a forward application of the analytical signal expression
    and the partial derivatives with respesct to each parameter of interest,
    as required by the abstract methods in the BaseModel.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the sequence related parametrs,
        e.g. TR, TE, TI, to fully describe the acquisitio process

    Attributes
    ----------
      TE : float
        Echo time (or any other timing valid in a bi-exponential fit).
      uk_scale : list of float
        Scaling factors for each unknown to balance the partial derivatives.
      guess : numpy.array, None
        The initial guess. Needs to be set using "computeInitialGuess"
        prior to fitting.
    """

    def __init__(self, par):
        super().__init__(par)
        self.TE = np.ones((self.NScan, 1, 1, 1))
        

        for i in range(self.NScan):
            self.TE[i, ...] = par["TEs"][i] * np.ones((1, 1, 1))

        par["unknowns_TGV"] = 3
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]
        
        self.init_M0 = None
        self.init_R2 = None
        self.init_B0 = None
        
        if "M0" in par["file"].keys():
            self.init_M0 = par["file"]["M0"][()]
        if "B0" in par["file"].keys():
            self.init_B0 = par["file"]["B0"][()]
        if "R2" in par["file"].keys():
            self.init_R2 = par["file"]["R2"][()]

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                1e100 / self.uk_scale[0], False))
        self.constraints.append(
            constraints(
                1 / self.uk_scale[1],
                150 / self.uk_scale[1],
                True))
       	self.constraints.append(
	    constraints(
		-1000 / self.uk_scale[2],
		1000 / self.uk_scale[2], True))
        
        self.guess = None
        self._ind1 = 0
        self._ind2 = 0


    def rescale(self, x):
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling factor and
        (applies a 1/x transformation for the time constants of the
        exponentials), yielding a result in milliseconds.

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
        tmp_x[0] = x[0, ...] * self.uk_scale[0]
        tmp_x[1] = x[1, ...] * self.uk_scale[1]
       	tmp_x[2] = x[2, ...] * self.uk_scale[2]

        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return {"data": tmp_x,
                "unknown_name": ["M0", "R2s", "B0"],
                "real_valued": const}

    def _execute_forward_3D(self, x):
        M0 = x[0, ...] * self.uk_scale[0]
        R2s = x[1, ...] * self.uk_scale[1]
       	B0 = x[2, ...] * self.uk_scale[2]

        S = M0 * np.exp(-self.TE * (R2s - 1j*B0))
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=self._DTYPE)
        return S*self.dscale

    def _execute_gradient_3D(self, x):
        M0 = x[0, ...]
        R2s = x[1, ...]
       	B0 = x[2, ...]
        
        grad_M0 = self.uk_scale[0] * np.exp(-self.TE * (
                R2s * self.uk_scale[1] - 1j*B0*self.uk_scale[2]))

        grad_R2s = -self.uk_scale[0] * M0 * \
            self.TE * self.uk_scale[1] * np.exp(-self.TE * (R2s * self.uk_scale[1] - 1j*B0*self.uk_scale[2]))
	
       	grad_B0 = self.uk_scale[0] * M0 * \
	    self.TE * self.uk_scale[2] * 1j * np.exp(-self.TE * (R2s * self.uk_scale[1] - 1j*B0*self.uk_scale[2]))

        grad = np.array([grad_M0, grad_R2s, grad_B0], dtype=self._DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad*self.dscale

    def computeInitialGuess(self, **kwargs):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting.

        Parameters
        ----------
          args : list of objects
            Serves as universal interface. No objects need to be passed
            here.
        """
        self.dscale = kwargs["dscale"]
        self.images = np.abs(kwargs["images"]/self.dscale)
        
        phase = np.angle(kwargs["images"])
        for j in range(phase.shape[0]):
            phase[j,:] = unwrap_phase(phase[j].squeeze())
                       
        mat_te = 1/(self.TE.squeeze()@self.TE.squeeze())*self.TE.squeeze()
        
        del_b0 = np.zeros(phase.shape[1:])
        for i in range(phase.shape[-3]):
            for j in range(phase.shape[-2]):
                for k in range(phase.shape[-1]):
                    del_b0[i,j,k] = mat_te@phase[:,i,j,k]
                    
        # import ipdb
        # import pyqmri
        # import matplotlib.pyplot as plt
        # ipdb.set_trace()
        
        
        # test_M0 = 1e-5 * np.ones(
        #     (self.NSlice, self.dimY, self.dimX),
        #     dtype=self._DTYPE)
        if self.init_M0 is not None:
            test_M0 = self.init_M0
        else:
            test_M0 = (kwargs["images"][0])/self.dscale
            
        if self.init_R2 is not None:
            test_R2s = self.init_R2
        else:
            test_R2s = 50 * np.ones(
                (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        if self.init_B0 is not None:
            test_B0 = self.init_B0
        else:
            test_B0 = del_b0

        self.guess = np.array([
            test_M0,
            test_R2s,
       	    test_B0], dtype=self._DTYPE)



    def plot_unknowns(self, x, dim_2D=False):
        unknowns = self.rescale(x)
        tmp_x = unknowns["data"]
        uknames = unknowns["unknown_name"]

        images = np.abs(self._execute_forward_3D(x) / self.dscale)
        images = np.reshape(images, self.TE.shape+images.shape[-3:])

        tmp_x[0] = np.abs(tmp_x[0])
        tmp_x[1:] = np.real(tmp_x[1:])
        tmp_x = np.real(tmp_x)

        if dim_2D:
            pass
        else:
            if not self._figure:
                self.ax = []
                plot_dim = int(np.ceil(np.sqrt(len(self.uk_scale))))
                plt.ion()
                self._figure = plt.figure(figsize=(12, 6))
                self._figure.subplots_adjust(hspace=0.3, wspace=0)
                wd_ratio = np.tile([1, 1 / 20, 1 / (5)], plot_dim)
                self.gs = gridspec.GridSpec(
                    plot_dim+1, 3*plot_dim,
                    width_ratios=wd_ratio, hspace=0.3, wspace=0)
                self._figure.tight_layout()
                self._figure.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs:
                    self.ax.append(plt.subplot(grid))
                    self.ax[-1].axis('off')
                self._plot = []
                for j in range(len(self.uk_scale)):
                    self._plot.append(
                        self.ax[3 * j].imshow(
                            tmp_x[j, int(self.NSlice / 2), ...]))
                    self.ax[3 *
                            j].set_title(uknames[j], color='white')
                    self.ax[3 * j + 1].axis('on')
                    cbar = self._figure.colorbar(
                        self._plot[j], cax=self.ax[3 * j + 1])
                    cbar.ax.tick_params(labelsize=12, colors='white')
                    for spine in cbar.ax.spines:
                        cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)
                self._figure.canvas.mpl_connect(
                    'button_press_event',
                    self.onclick)

                self.plot_ax = plt.subplot(self.gs[-1, :])
                self.plot_ax.set_title("Time course", color='w')
                self.time_course_ref = []

                self.time_course_ref.append(self.plot_ax.plot(
                    self.TE.squeeze(), np.real(
                        self.images[...,
                                    int(self.NSlice/2),
                                    self._ind2, self._ind1]).T,
                    'x')[0])
                
                self.plot_ax.set_prop_cycle(None)
                legend = self.plot_ax.legend(frameon=True, framealpha=0.3)
                for _txt in legend.texts:
                    _txt.set_alpha(0.3)
                for lh in legend.legendHandles:
                    lh._legmarker.set_alpha(0.3)
                    
                self.time_course = self.plot_ax.plot(
                    self.TE.squeeze(), np.real(
                        images[..., int(self.NSlice/2),
                               self._ind2, self._ind1]).squeeze())
                self.plot_ax.set_ylim(
                    np.minimum(np.real(images[...,
                                              int(self.NSlice/2),
                                              self._ind2,
                                              self._ind1]).min(),
                               np.real(self.images[...,
                                                   int(self.NSlice/2),
                                                   self._ind2,
                                                   self._ind1]).min()),
                    1.2*np.maximum(np.real(images[...,
                                                  int(self.NSlice/2),
                                                  self._ind2,
                                                  self._ind1]).max(),
                                   np.real(self.images[...,
                                                       int(self.NSlice/2),
                                                       self._ind2,
                                                       self._ind1]).max()))
                for spine in self.plot_ax.spines:
                    self.plot_ax.spines[spine].set_color('white')
                self.plot_ax.xaxis.label.set_color('white')
                self.plot_ax.yaxis.label.set_color('white')
                self.plot_ax.tick_params(axis='both', colors='white')

                plt.draw()
                plt.show()
                plt.pause(1e-4)
            else:
                for j in range(len(self.uk_scale)):
                    self._plot[j].set_data(
                        tmp_x[j, int(self.NSlice / 2), ...])
                    self._plot[j].set_clim(
                        [tmp_x[j].min(), tmp_x[j].max()])


                self.time_course[0].set_ydata(
                    np.real(images[
                        ..., int(self.NSlice/2), self._ind2, self._ind1]))
                
                self.plot_ax.set_ylim(
                    np.minimum(np.real(images[...,
                                              int(self.NSlice/2),
                                              self._ind2,
                                              self._ind1]).min(),
                               np.real(self.images[...,
                                                   int(self.NSlice/2),
                                                   self._ind2,
                                                   self._ind1]).min()),
                    1.2*np.maximum(np.real(images[...,
                                                  int(self.NSlice/2),
                                                  self._ind2,
                                                  self._ind1]).max(),
                                   np.real(self.images[...,
                                                       int(self.NSlice/2),
                                                       self._ind2,
                                                       self._ind1]).max()))
                plt.draw()
                plt.pause(1e-10)

    def onclick(self, event):
        if event.inaxes in self.ax[::3]:
            self._ind1 = int(event.xdata)
            self._ind2 = int(event.ydata)

            self.time_course_ref[0].set_ydata(np.real(
                    self.images[...,
                                int(self.NSlice/2),
                                self._ind2, self._ind1]).T)