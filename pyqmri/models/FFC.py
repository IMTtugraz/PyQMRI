#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints, DTYPE
plt.ion()



class Model(BaseModel):
    def __init__(self, par):
        super().__init__(par)
        self.constraints = []
        self.t = par["t"]
        self.b = par["b"]

        if len(self.t.shape)<2:
            self.b = self.b[None,:]
            self.t = self.t[None,:]

        self.numT1Scale = len(self.b)-1

        par["unknowns_TGV"] = 2 + len(self.b)
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]

        self.unknowns = par["unknowns"]

        for j in range(par["unknowns"]):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(0,
                        100,
                        False))
        self.constraints.append(
            constraints(0,
                        100,
                        False))
        self.constraints.append(
            constraints(1/1000,
                        1/5,
                        True))
        for j in range(self.numT1Scale):
            self.constraints.append(
                constraints(5*(j+1)/2000,
                            1/5,
                            True))

    def rescale(self, x):
        tmp_x = np.copy(x)
        tmp_x[0] *= self.uk_scale[0]
        tmp_x[1] *= self.uk_scale[1]
        tmp_x[2] = 1/(x[2] * self.uk_scale[2])
        for j in range(self.numT1Scale):
            tmp_x[3+j] = 1/(x[3+j] * self.uk_scale[3+j])
        return tmp_x

    def _execute_forward_2D(self, x, islice):
        pass

    def _execute_gradient_2D(self, x, islice):
        pass

    def _execute_forward_3D(self, x):
        S = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        t = self.t[0][:, None, None, None]
        S[:len(t)] = (
            -x[0] * self.uk_scale[0]
            * np.exp(-t * (x[2] * self.uk_scale[2]))
            + (1 - np.exp(-t * (x[2] * self.uk_scale[2])))
            * x[1] * self.uk_scale[1]*self.b[0])
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            S[offset*(j+1):offset*(j+2)] = (
                -x[0] * self.uk_scale[0]
                * np.exp(-t
                         * (x[3+j] * self.uk_scale[3+j]))
                + (1 - np.exp(-t
                              * (x[3+j] * self.uk_scale[3+j])))
                * x[1] * self.uk_scale[1]*self.b[1+j]
                )
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):

        gradM0 = self._gradM0(x)
        gradXi = self._gradXi(x)
        gradR1 = self._gradR1(x)
        gradCx = np.zeros(
            (self.numT1Scale,
             self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        for j in range(self.numT1Scale):
            self._gradCx(gradCx, x, j)
        gradCx[~np.isfinite(gradCx)] = 1e-20

        grad = np.concatenate(
            (np.array([gradM0, gradXi, gradR1], dtype=DTYPE),
             gradCx), axis=0)
        return grad

    def _gradM0(self, x):
        grad = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        t = self.t[0][:, None, None, None]
        grad[:len(t)] = (
            - self.uk_scale[0]
            * np.exp(-t * (x[2] * self.uk_scale[2]))
            )
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            grad[offset*(j+1):offset*(j+2)] = (
                -self.uk_scale[0]
                * np.exp(-t
                         * (x[3+j] * self.uk_scale[3+j]))
                )
        grad[~np.isfinite(grad)] = 1e-20

        return grad

    def _gradXi(self, x):
        grad = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        t = self.t[0][:, None, None, None]
        grad[:len(t)] = (
            (1 - np.exp(-t * (x[2] * self.uk_scale[2])))
            * self.uk_scale[1]*self.b[0])
        for j in range(self.numT1Scale):
            offset = len(self.t[j+1])
            t = self.t[j+1][:, None, None, None]
            grad[offset*(j+1):offset*(j+2)] = (
                (1 - np.exp(-t
                            * (x[3+j] * self.uk_scale[3+j])))
                * self.uk_scale[1]*self.b[1+j]
                )
        grad[~np.isfinite(grad)] = 1e-20

        return grad

    def _gradR1(self, x):
        grad = np.zeros(
            (self.NScan, self.NSlice, self.dimY, self.dimX),
            dtype=DTYPE)
        t = self.t[0][:, None, None, None]
        grad[:len(t)] = (
            x[0]*self.uk_scale[0]
            * self.uk_scale[2]
            * t
            * np.exp(-t * (x[2] * self.uk_scale[2]))
            + 1
            * self.uk_scale[2]
            * self.b[0] * t
            * x[1] * self.uk_scale[1]
            * np.exp(-t * (x[2] * self.uk_scale[2]))
            )
        # for j in range(self.numT1Scale):
        #     offset = len(self.t[j+1])
        #     t = self.t[j+1][:, None, None, None]
        #     grad[offset*(j+1):offset*(j+2)] = (
        #         - x[0] * self.uk_scale[0]
        #         / (x[3+j]**2 * self.uk_scale[3+j])
        #         * t
        #         * np.exp(-t
        #                  / (x[3+j] * self.uk_scale[3+j]))
        #         - self.b[1+j] * t
        #         / (x[3+j]**2 * self.uk_scale[3+j])
        #         * x[1] * self.uk_scale[1]
        #         * np.exp(-t
        #                  / (x[3+j] * self.uk_scale[3+j]))
        #         )
        grad[~np.isfinite(grad)] = 1e-20

        return grad

    def _gradCx(self, grad, x, ind):
        offset = len(self.t[ind+1])
        t = self.t[ind+1][:, None, None, None]
        grad[ind, (ind+1)*offset:(ind+2)*offset] = (
            x[0]*self.uk_scale[0]
            * self.uk_scale[3+ind]
            * t
            * np.exp(-t
                     * (x[3+ind] * self.uk_scale[3+ind]))
            + self.b[1+ind] * t
            * self.uk_scale[3+ind]
            * x[1] * self.uk_scale[1]
            * np.exp(- t
                     * (x[3+ind] * self.uk_scale[3+ind]))
            )

    def plot_unknowns(self, x, dim_2D=False):
        images = np.abs(self._execute_forward_3D(x) / self.dscale)[
            :self.t.shape[1]]
        tmp_x = (self.rescale(x))
        tmp_x[0:2] = np.abs(tmp_x[0:2])
        # tmp_x[1] /= self.dscale
        tmp_x = np.real(tmp_x)

        ind1 = 26
        ind2 = 26

        if dim_2D:
            pass
        else:
            self.ax = []
            if not self.figure:
                plot_dim = int(np.ceil(np.sqrt(len(self.uk_scale))))
                plt.ion()
                self.figure = plt.figure(figsize=(12, 6))
                self.figure.subplots_adjust(hspace=0.3, wspace=0)
                wd_ratio = np.tile([1, 1 / 20, 1 / (5)], plot_dim)
                self.gs = gridspec.GridSpec(
                    plot_dim+1, 3*plot_dim,
                    width_ratios=wd_ratio, hspace=0.3, wspace=0)
                self.figure.tight_layout()
                self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
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
                    cbar = self.figure.colorbar(
                        self._plot[j], cax=self.ax[3 * j + 1])
                    cbar.ax.tick_params(labelsize=12, colors='white')
                    for spine in cbar.ax.spines:
                        cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)

                self.plot_ax = plt.subplot(self.gs[-1, :])

                self.time_course_ref = self.plot_ax.scatter(
                    self.t[0], np.real(
                        self.images[:, int(self.NSlice/2), ind2, ind1]),
                    color='g', marker="2")
                self.time_course = self.plot_ax.plot(
                    self.t[0], np.real(
                        images[:, int(self.NSlice/2), ind2, ind1]), 'r')[0]
                self.plot_ax.set_ylim(
                    np.real(self.images[:,
                                        int(self.NSlice/2),
                                        ind2,
                                        ind1]).min() - np.real(
                            self.images[:,
                                        int(self.NSlice/2),
                                        ind2,
                                        ind1]).min() * 0.01,
                    np.real(self.images[:,
                                        int(self.NSlice/2),
                                        ind2,
                                        ind1]).max() + np.real(
                           self.images[:,
                                       int(self.NSlice/2),
                                       ind2,
                                       ind1]).max() * 0.01)
                for spine in self.plot_ax.spines:
                    self.plot_ax.spines[spine].set_color('white')
                plt.draw()
                plt.show()
                plt.pause(1e-4)

            else:
                for j in range(len(self.uk_scale)):
                    self._plot[j].set_data(tmp_x[j, int(self.NSlice / 2), ...])
                    self._plot[j].set_clim([tmp_x[j].min(), tmp_x[j].max()])
                self.time_course.set_ydata(
                    np.real(images[:, int(self.NSlice/2), ind2, ind1]))
                self.plot_ax.set_ylim(
                    np.real(self.images[:,
                                        int(self.NSlice/2),
                                        ind2,
                                        ind1]).min() - np.real(
                            self.images[:,
                                        int(self.NSlice/2),
                                        ind2,
                                        ind1]).min() * 0.01,
                    np.real(self.images[:,
                                        int(self.NSlice/2),
                                        ind2,
                                        ind1]).max() + np.real(
                           self.images[:,
                                       int(self.NSlice/2),
                                       ind2,
                                       ind1]).max() * 0.01)
                plt.draw()
                plt.pause(1e-10)

    def computeInitialGuess(self, *args):
        self.dscale = args[1]
        self.images = np.abs(args[0]/args[1])[
            :self.t.shape[1]]
        test_M0 = 1*np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        self.constraints[0].update(1/args[1])
        test_Xi = 1*np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        # self.constraints[1].update(1/args[1])
        test_R1 = 1/400 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_Cx = []
        # self.b *= args[1]
        for j in range(self.numT1Scale):
            test_Cx.append(1/(200/(j+1)) *
                np.ones(
                    (self.NSlice, self.dimY, self.dimX), dtype=DTYPE))
        self.guess = np.array(
            [test_M0, test_Xi, test_R1] + test_Cx, dtype=DTYPE)
