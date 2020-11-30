#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints
plt.ion()



class Model(BaseModel):
    def __init__(self, par):
        super().__init__(par)
        self.constraints = []
        self.t = par["t"]
        self.b = par["b"]
        if "b0" in par.keys():
            self.b0 = par["b0"]
        else:
            self.b0 = self.b[0]

        if len(self.t.shape) < 2:
            self.b = self.b[None]
            self.t = self.t[None]

        self.numT1Scale = len(self.b)-1

        par["unknowns_TGV"] = 2 + len(self.b) + len(self.b) - 1
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        self.unknowns = par["unknowns"]

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(-1000,
                        1000,
                        False))
        self.constraints.append(
            constraints(0,
                        2,
                        False))
        self.constraints.append(
            constraints(1,
                        2000,
                        True))
        for j in range(self.numT1Scale):
            self.constraints.append(
                constraints(1,
                            2000,
                            True))
        for j in range(self.numT1Scale):
            self.constraints.append(
                constraints(0,
                            2,
                            False))
        self._ind1 = 0
        self._ind2 = 0
        self._labels = []
        for j in range(len(self.b)):
            self._labels.append(
                "Field "+str(np.round(self.b[j]*1e3, 2))+" mT")
            
            
    def rescale(self, x):
        tmp_x = np.copy(x)
        tmp_x[0] *= self.uk_scale[0]
        tmp_x[1] *= self.uk_scale[1]
        tmp_x[2] *= self.uk_scale[2]
        ukname = ["C", "alpha", "T1_1"]
        for j in range(self.numT1Scale):
            tmp_x[3+j] *= self.uk_scale[3+j]
            ukname.append("T1_"+str(2+j))
        for j in range(self.numT1Scale):
            tmp_x[3+self.numT1Scale+j] *= self.uk_scale[3+self.numT1Scale+j]
            ukname.append("al_"+str(2+j))
        const = []
        for constrained in self.constraints:
            const.append(constrained.real)
        return {"data": tmp_x,
                "unknown_name": ukname,
                "real_valued": const}

    def _execute_forward_2D(self, x, islice):
        pass

    def _execute_gradient_2D(self, x, islice):
        pass

    def _execute_forward_3D(self, x):
        S = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        t = self.t[0][:, None, None, None]
        S[:len(t)] = (
            x[0] * self.uk_scale[0]
            * (-x[1]*self.uk_scale[1]*self.b0 *
               np.exp(-t / (x[2] * self.uk_scale[2]))
               + (1 - np.exp(-t / (x[2] * self.uk_scale[2])))
               * self.b[0])
            )
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            S[offset*(j+1):offset*(j+2)] = (
                x[0] * self.uk_scale[0]
                * (-x[3+self.numT1Scale+j]*self.uk_scale[3+self.numT1Scale+j]*self.b0 *
                   np.exp(-t / (x[3+j] * self.uk_scale[3+j]))
                   + (1 - np.exp(-t / (x[3+j] * self.uk_scale[3+j])))
                   * self.b[1+j])
                )
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=self._DTYPE)
        return S

    def _execute_gradient_3D(self, x):

        gradC = self._gradC(x)
        gradAlpha = self._gradAlpha(x)
        gradT1 = self._gradT1(x)

        grad = np.concatenate(
            (np.array([gradC, gradAlpha[0]], dtype=self._DTYPE),
             gradT1), axis=0)

        grad = np.concatenate((grad, gradAlpha[1:]), axis=0)
        return grad

    def _gradC(self, x):
        grad = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        t = self.t[0][:, None, None, None]
        grad[:len(t)] = (
            self.uk_scale[0]
            * (-x[1]*self.uk_scale[1]*self.b0 *
               np.exp(-t / (x[2] * self.uk_scale[2]))
               + (1 - np.exp(-t / (x[2] * self.uk_scale[2])))
               * self.b[0])
            )
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            grad[offset*(j+1):offset*(j+2)] = (
                self.uk_scale[0]
                * (-x[3+self.numT1Scale+j]*self.uk_scale[3+self.numT1Scale+j]*self.b0 *
                   np.exp(-t / (x[3+j] * self.uk_scale[3+j]))
                   + (1 - np.exp(-t / (x[3+j] * self.uk_scale[3+j])))
                   * self.b[1+j])
                )
        grad[~np.isfinite(grad)] = 1e-20

        return grad

    def _gradAlpha(self, x):
        grad = np.zeros(
            (self.numT1Scale+1, self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        t = self.t[0][:, None, None, None]
        grad[0, :len(t)] = (
            x[0] * self.uk_scale[0]
            * (-self.uk_scale[1]*self.b0 *
               np.exp(-t / (x[2] * self.uk_scale[2]))
               )
            )
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            grad[j+1, offset*(j+1):offset*(j+2)] = (
                x[0] * self.uk_scale[0]
                * (-self.uk_scale[3+self.numT1Scale+j]*self.b0 *
                   np.exp(-t / (x[3+j] * self.uk_scale[3+j]))
                   )
                )
        grad[~np.isfinite(grad)] = 1e-20

        return grad

    def _gradT1(self, x):
        grad = np.zeros(
            (self.numT1Scale+1, self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=self._DTYPE)
        t = self.t[0][:, None, None, None]
        grad[0, :len(t)] = (
            x[0]*self.uk_scale[0]*(
                -x[1]*self.uk_scale[1]*self.b0*t
                * np.exp(-t/(x[2]*self.uk_scale[2]))
                - self.b[0]*t
                * np.exp(-t/(x[2]*self.uk_scale[2]))
                )/(x[2]**2*self.uk_scale[2])
            )
        for ind in range(self.numT1Scale):
            offset = len(self.t[ind+1])
            t = self.t[ind+1][:, None, None, None]
            grad[ind+1, (ind+1)*offset:(ind+2)*offset] = (
                x[0]*self.uk_scale[0]*(
                    -x[3+self.numT1Scale+ind]*self.uk_scale[3+self.numT1Scale+ind]*self.b0*t
                    * np.exp(-t/(x[3+ind]*self.uk_scale[3+ind]))
                    - self.b[1+ind]*t
                    * np.exp(-t/(x[3+ind]*self.uk_scale[3+ind]))
                    )
                )/(x[3+ind]**2*self.uk_scale[3+ind])
        grad[~np.isfinite(grad)] = 1e-20

        return grad
    def plot_unknowns(self, x, dim_2D=False):
        unknowns = self.rescale(x)
        tmp_x = unknowns["data"]

        images = np.abs(self._execute_forward_3D(x) / self.dscale)
        images = np.reshape(images, self.t.shape+images.shape[-3:])

        tmp_x[0] = np.abs(tmp_x[0])/self.dscale
        tmp_x[1] = np.abs(tmp_x[1])
        tmp_x[-self.numT1Scale:] = np.abs(tmp_x[-self.numT1Scale:])
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
                            j].set_title('UK : ' +
                                         str(j), color='white')
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
                for j in range(len(self.b)):
                    self.time_course_ref.append(self.plot_ax.plot(
                        self.t[j], np.real(
                            self.images[j, :,
                                        int(self.NSlice/2),
                                        self._ind2, self._ind1]).T,
                        'x', label=self._labels[j])[0])
                self.plot_ax.set_prop_cycle(None)
                legend = self.plot_ax.legend(frameon=True, framealpha=0.3)
                for _txt in legend.texts:
                    _txt.set_alpha(0.3)
                for lh in legend.legendHandles:
                    lh._legmarker.set_alpha(0.3)
                self.time_course = self.plot_ax.plot(
                    self.t.T, np.real(
                        images[..., int(self.NSlice/2),
                               self._ind2, self._ind1]).T)
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

                for j in range(len(self.b)):
                    self.time_course[j].set_ydata(
                        np.real(images[
                            j, :, int(self.NSlice/2), self._ind2, self._ind1]))
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
            for j in range(len(self.b)):
                self.time_course_ref[j].set_ydata(np.real(
                        self.images[j, :,
                                    int(self.NSlice/2),
                                    self._ind2, self._ind1]).T)

    def computeInitialGuess(self, *args):
        self.dscale = args[1]
        self.images = np.reshape(np.abs(args[0]/args[1]),
                                 self.t.shape+args[0].shape[-3:])
        test_M0 = 0.1*np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        self.constraints[0].update(1/args[1])
        test_Xi = 1*np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        # self.constraints[1].update(1/args[1])
        test_R1 = 300 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        test_Cx = []
        # self.b *= args[1]
        for j in range(self.numT1Scale):
            test_Cx.append(
                300/(j+1) *
                np.ones(
                    (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE))
        test_Ca = []
        # self.b *= args[1]
        for j in range(self.numT1Scale):
            test_Ca.append(
                1 *
                np.ones(
                    (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE))

        self.guess = np.array(
            [test_M0, test_Xi, test_R1] + test_Cx + test_Ca, dtype=self._DTYPE)
