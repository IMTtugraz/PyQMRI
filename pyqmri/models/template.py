#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the template base class model."""
import itertools
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class constraints:
    """Constraints for a parameter.

    This class holds min/max and real value constrains for a parameter. It
    supports updating these based on the current estimated sclaing between
    each partial derivative.

    Parameters
    ----------
      min_val : float, -numpy.inf
        The minimum value.
      max_val : float, numpy.inf
        The maximum value.
      real_const : bool, false
        Constrain to real values (true) or complex values (false).

    Attributes
    ----------
      constraints : list of pyqmri.models.template.constrains
        An empy list of constrains objects.
      NScan : int
        Number of scans (dynamics).
      NSlice : int
        Number of slices.
      dimX, dimY : int
        The image dimensions.
      figure : matplotlib.pyplot.figure, None
        The placeholder figure object
    """

    def __init__(self, min_val=-np.inf, max_val=np.inf,
                 real_const=False):
        self.min = min_val
        self.max = max_val
        self.real = real_const

    def update(self, scale):
        """Update the constrains based on current scaling factor.

        Parameters
        ----------
          scale : float
            The new scaling factor which should be used.
        """
        self.min = self.min / scale
        self.max = self.max / scale


class BaseModel(ABC):
    """Base model for MRI parameter quantification.

    This class holds the base model to derive other signal models from.
    It defines abstract a forward application of the analytical signal
    expression    and the partial derivatives with respesct to each parameter
    of interest.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the sequence related parametrs,
        e.g. Number of Scans, Slices, and image dimension.

    Attributes
    ----------
      constraints : list of pyqmri.models.template.constrains
        An empy list of constrains objects.
      NScan : int
        Number of scans (dynamics).
      NSlice : int
        Number of slices.
      dimX, dimY : int
        The image dimensions.
    """

    def __init__(self, par):
        super().__init__()
        self.constraints = []
        self.uk_scale = []
        self.NScan = par["NScan"]
        self.NSlice = par["NSlice"]
        self.dimX = par["dimX"]
        self.dimY = par["dimY"]
        self._DTYPE = par["DTYPE"]
        self._DTYPE_real = par["DTYPE_real"]
        self._figure = None
        self._plot_trans = []
        self._plot_cor = []
        self._plot_sag = []

    def rescale(self, x):
        """Rescale the unknowns with the scaling factors.

        Rescales each unknown with the corresponding scaling factor.

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
        uk_names = []
        for i in range(x.shape[0]):
            tmp_x[i] *= self.uk_scale[i]
            uk_names.append("Unkown_"+str(i))
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return {"data": tmp_x,
                "unknown_name": uk_names,
                "real_valued": const}

    def execute_forward(self, x, islice=None):
        """Execute the signal model from parameter to imagespace.

        This function exectues the given signal model to generate an image
        series, given estimated parameters.

        Parameters
        ----------
          x : numpy.array
            The array of quantitative parameters to be fitted
          islice : int, None
            Currently unused.
        """
        # if islice is None:
        return self._execute_forward_3D(x)

    def execute_gradient(self, x, islice=None):
        """Execute the partial derivatives of the signal model.

        This function exectues the partial derivatives with respect to each
        unknown parameter, based on the signal model.

        Parameters
        ----------
          x : numpy.array
            The array of quantitative parameters to be fitted
          islice : int, None
            Currently unused.
        """
        # if islice is None:
        return self._execute_gradient_3D(x)

    @abstractmethod
    def _execute_forward_3D(self, x):
        ...

    @abstractmethod
    def _execute_gradient_3D(self, x):
        ...

    def plot_unknowns(self, x):
        """Plot the unkowns in an interactive figure.

        This function can be used to plot intermediate results during the
        optimization process.

        Parameters
        ----------
          x : dict
            A Python dictionary containing the array of unknowns to be
            displayed, the associated names and real value constrains.
        """
        unknowns = self.rescale(x)
        NSlice = unknowns["data"].shape[1]
        numunknowns = unknowns["data"].shape[0]
        [dimZ, dimY, dimX] = unknowns["data"].shape[1:]

        if not self._figure:
            self._ax_trans = []
            self._ax_sag = []
            self._ax_cor = []
            plot_dim_x = int(np.ceil(np.sqrt(numunknowns)))
            plot_dim_y = int(np.round(np.sqrt(numunknowns)))
            plt.ion()
            self._figure = plt.figure(figsize=(12, 6))
            self._figure.subplots_adjust(hspace=0.3, wspace=0)
            wd_ratio = np.tile([1, dimZ/dimX, 1 / 20, 1 / (5)], plot_dim_x)
            hd_ratio = np.tile([1, dimZ/dimY, 1 / (5)], plot_dim_y)
            gs = gridspec.GridSpec(
                3 * plot_dim_y, 4 * plot_dim_x,
                width_ratios=wd_ratio,
                height_ratios=hd_ratio,
                hspace=0,
                wspace=0)
            gs.tight_layout(self._figure)
            self._figure.patch.set_facecolor(plt.cm.viridis.colors[0])

            for i, j in itertools.product(
                    range(plot_dim_y),
                    range(plot_dim_x)):
                if i*plot_dim_x+j < numunknowns:
                    ax_trans = plt.subplot(gs[3*i, 4*j])
                    ax_sag = plt.subplot(gs[3*i, 4*j+1])
                    ax_cor = plt.subplot(gs[3*i+1, 4*j])
                    ax_bar = plt.subplot(gs[3*i:3*i+2, 4*j+2])
                    ax_trans.axis('off')
                    ax_sag.axis('off')
                    ax_cor.axis('off')

                    ax_trans.set_anchor('SE')
                    ax_sag.set_anchor('SW')
                    ax_cor.set_anchor('NE')
                    
                    self._ax_trans.append(ax_trans)
                    self._ax_sag.append(ax_sag)
                    self._ax_cor.append(ax_cor)

                    if unknowns["real_valued"][i*plot_dim_x+j]:
                        def mytrafo(x):
                            return np.real(x)
                    else:
                        def mytrafo(x):
                            return np.abs(x)

                    self._plot_trans.append(
                        ax_trans.imshow(
                            mytrafo(unknowns["data"][i*plot_dim_x+j,
                                                     int(NSlice / 2), ...])))
                    self._ax_trans[-1].volume = unknowns["data"][
                        i*plot_dim_x+j]
                    self._ax_trans[-1].index = int(NSlice / 2)
                    
                    self._plot_cor.append(
                        ax_cor.imshow(
                            mytrafo(unknowns["data"][i*plot_dim_x+j,
                                                     ..., int(dimY / 2), :])))
                    self._ax_cor[-1].volume = np.swapaxes(
                            unknowns["data"][i*plot_dim_x+j], 0, 1)
                    self._ax_cor[-1].index = int(dimY / 2)
                    
                    self._plot_sag.append(
                        ax_sag.imshow(
                            mytrafo(unknowns["data"][i*plot_dim_x+j,
                                                     ..., int(dimX / 2)].T)))
                    self._ax_sag[-1].volume = unknowns["data"][
                        i*plot_dim_x+j].T
                    self._ax_sag[-1].index = int(dimX / 2)
                    
                    ax_trans.set_title(
                        unknowns["unknown_name"][i*plot_dim_x+j],
                        color='white')

                    cbar = self._figure.colorbar(
                        self._plot_trans[-1], cax=ax_bar)
                    cbar.ax.tick_params(labelsize=12, colors='white')
                    for spine in cbar.ax.spines:
                        cbar.ax.spines[spine].set_color('white')
            plt.draw()
            plt.pause(1e-10)
            
            self._figure.canvas.mpl_connect(
                'scroll_event',
                self.onscroll)

        else:
            for j in range(numunknowns):
                if unknowns["real_valued"][j]:
                    def mytrafo(x):
                        return np.real(x)
                else:
                    def mytrafo(x):
                        return np.abs(x)
                    
                self._ax_trans[j].volume = mytrafo(unknowns["data"][j])
                self._ax_cor[j].volume = mytrafo(np.swapaxes(
                    unknowns["data"][j], 0, 1))
                self._ax_sag[j].volume = mytrafo(unknowns["data"][j].T)
                
                self._ax_trans[j].images[0].set_array(
                    self._ax_trans[j].volume[self._ax_trans[j].index])
                self._ax_cor[j].images[0].set_array(
                    self._ax_cor[j].volume[self._ax_cor[j].index])
                self._ax_sag[j].images[0].set_array(
                    self._ax_sag[j].volume[self._ax_sag[j].index])
                
                minval = mytrafo(unknowns["data"][j]).min()
                maxval = mytrafo(unknowns["data"][j]).max()
                
                self._plot_trans[j].set_clim([minval, maxval])
                self._plot_cor[j].set_clim([minval, maxval])
                self._plot_sag[j].set_clim([minval, maxval])
            plt.draw()
            plt.pause(1e-10)
            
    def onscroll(self, event):
        if event.inaxes in self._ax_trans:
            fig = event.canvas.figure
            ax = self._ax_trans
        
        elif event.inaxes in self._ax_cor:
            fig = event.canvas.figure
            ax = self._ax_cor
                        
        elif event.inaxes in self._ax_sag:
            fig = event.canvas.figure
            ax = self._ax_sag
        else:
            return
        
        for i, axes in enumerate(ax):
            if axes.index is not None:
                volume = axes.volume
                if (int((axes.index - event.step) >= volume.shape[0]) or
                        int((axes.index - event.step) < 0)):
                    pass
                else:
                    ax[i].index = int((axes.index - event.step) % volume.shape[0])
                    ax[i].images[0].set_array(volume[ax[i].index])
                    fig.canvas.draw()
        plt.draw()
        plt.pause(1e-10)

    @abstractmethod
    def computeInitialGuess(self, *args):
        """Initialize unknown array for the fitting.

        This function provides an initial guess for the fitting.
        """
        ...
