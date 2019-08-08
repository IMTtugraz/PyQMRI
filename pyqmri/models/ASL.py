#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints, DTYPE
import numexpr as ne
plt.ion()
unknowns_TGV = 2
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        self.constraints = []
        full_slices = par["file"]["T1b"].shape[0]
        sliceind = slice(int(full_slices / 2) -
                         int(np.floor((par["NSlice"]) / 2)),
                         int(full_slices / 2) +
                         int(np.ceil(par["NSlice"] / 2)))
        self.T1b = par["file"]["T1b"][sliceind]
        self.T1 = par["file"]["T1"][sliceind]
        self.lambd = par["file"]["lambd"][sliceind]
        self.M0 = par["file"]["M0"][sliceind]
        self.tau = par["file"]["tau"][:, sliceind]
        self.t = par['t']
        self.alpha = par["file"]["alpha"][sliceind]

        self.NScan = par["NScan"]
        self.NSlice = par["NSlice"]
        self.dimY = par["dimY"]
        self.dimX = par["dimX"]
        self.unknowns = unknowns_TGV+unknowns_H1
        self.images = images
        self.dscale = par["dscale"]

        for j in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)
        self.guess = self._set_init_scales(images)

        self.constraints.append(
            constraints(0,
                        200 * self.dscale,
                        True))
        self.constraints.append(
            constraints(0,
                        4/60,
                        True))

    @staticmethod
    def _expAttT1b(del_t, del_t_sc, T1b):
        return ne.evaluate(
            "exp(-(del_t*del_t_sc)/T1b)")

    @staticmethod
    def _T1pr(T1, f, f_sc, lambd):
        return ne.evaluate(
            "1/T1+f*f_sc/lambd")

    @staticmethod
    def _S1(M0, alpha, lambd, f, T1, T1p, del_t,  t, tau, expAttT1b):
        return ne.evaluate(
            "2*alpha*M0/lambd * f/T1p * expAttT1b * \
            (1-exp(-(t-del_t) * T1p))")

    @staticmethod
    def _S2(M0, alpha, lambd, f, T1, T1p, del_t, t, tau, expAttT1b):
        return ne.evaluate(
            "2*alpha*M0/lambd * f/T1p * expAttT1b * \
             exp(-(t-del_t-tau) * T1p) * \
             (1-exp(-tau * T1p))")

    @staticmethod
    def _delCBF1(M0, alpha, lambd, f, f_sc, T1, del_t,
                 del_t_sc, T1b, t, tau, T1p, expAttT1b):
        return ne.evaluate(
            "(-2*M0*f*f_sc**2*(del_t*del_t_sc - t) *\
             exp((del_t*del_t_sc - t) * T1p) * expAttT1b / (lambd**2*T1p) -\
             2*M0*f*f_sc**2 * (-exp((del_t*del_t_sc - t) * T1p) + 1) * \
             expAttT1b / (lambd**2*T1p**2) +\
             2*M0*f_sc*(-exp((del_t*del_t_sc - t) * T1p) + 1) * expAttT1b /\
             (lambd*T1p))*alpha")

    @staticmethod
    def _delCBF2(M0, alpha, lambd, f, f_sc, T1, del_t,
                 del_t_sc, T1b, t, tau, T1p, expAttT1b):
        return ne.evaluate(
            "(2*M0*f*f_sc**2*tau *  exp(-tau*T1p) *\
              exp(T1p * (del_t*del_t_sc - t + tau)) * \
              expAttT1b / (lambd**2*T1p) +\
              2*M0*f*f_sc**2 * (1 - exp(-tau*T1p)) * \
              (del_t*del_t_sc - t + tau) *\
              exp(T1p * (del_t*del_t_sc - t + tau)) * \
              expAttT1b / (lambd**2*T1p) -\
              2*M0*f*f_sc**2 * (1 - exp(-tau*T1p)) * \
              exp(T1p * (del_t*del_t_sc - t + tau)) * \
              expAttT1b / (lambd**2*T1p**2) +\
              2*M0*f_sc * (1 - exp(-tau*T1p)) * \
              exp(T1p * (del_t*del_t_sc - t + tau)) * \
              expAttT1b / (lambd*T1p)) *\
              alpha")

    @staticmethod
    def _delATT1(M0, alpha, lambd, f, f_sc, T1, del_t,
                 del_t_sc, T1b, t, tau, T1p, expAttT1b):
        return ne.evaluate(
            "(-2*M0*del_t_sc*f*f_sc * exp((del_t*del_t_sc - t) * T1p) *\
              expAttT1b/lambd -\
              2*M0*del_t_sc*f*f_sc*(- exp((del_t*del_t_sc - t) * T1p) + 1) *\
              expAttT1b / (T1b*lambd*T1p))*alpha")

    @staticmethod
    def _delATT2(M0, alpha, lambd, f, f_sc, T1, del_t,
                 del_t_sc, T1b, t, tau, T1p, expAttT1b):
        return ne.evaluate(
            "(2*M0*del_t_sc*f*f_sc * (1 - exp(-tau*T1p)) *\
              exp(T1p * (del_t*del_t_sc - t + tau)) * expAttT1b/lambd -\
              2*M0*del_t_sc*f*f_sc * (1 - exp(-tau*T1p)) *\
              exp(T1p * (del_t*del_t_sc - t + tau)) * expAttT1b /\
              (T1b*lambd*T1p))*alpha")

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        f = x[0, ...] * self.uk_scale[0]
        del_t = x[1, ...] * self.uk_scale[1]

        S = np.zeros((self.NScan, self.NSlice, self.dimY, self.dimX),
                     dtype=DTYPE)

        T1prinv = Model._T1pr(self.T1, x[0], self.uk_scale[0], self.lambd)
        expAtt = Model._expAttT1b(x[1], self.uk_scale[1], self.T1b)
        for j in range((self.t).size):
            ind_low = self.t[j] >= del_t
            ind_high = self.t[j] < (del_t+self.tau[j])
            ind = ind_low & ind_high
            if np.any(ind):
                S[j, ind] = Model._S1(self.M0[ind], self.alpha[ind],
                                      self.lambd[ind], f[ind], self.T1[ind],
                                      T1prinv[ind], del_t[ind],
                                      self.t[j], self.tau[j, ind], expAtt[ind])
            ind = self.t[j] >= del_t + self.tau[j]
            if np.any(ind):
                S[j, ind] = Model._S2(self.M0[ind], self.alpha[ind],
                                      self.lambd[ind], f[ind], self.T1[ind],
                                      T1prinv[ind], del_t[ind],
                                      self.t[j], self.tau[j, ind], expAtt[ind])
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        f_sc = self.uk_scale[0]
        del_t_sc = self.uk_scale[1]
        del_t = x[1]*del_t_sc
        grad = np.zeros((self.unknowns, self.NScan,
                         self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        t = self.t
        T1prinv = Model._T1pr(self.T1, x[0], self.uk_scale[0], self.lambd)
        expAtt = Model._expAttT1b(x[1], self.uk_scale[1], self.T1b)
        for j in range((self.t).size):
            ind_low = self.t[j] >= del_t
            ind_high = self.t[j] < (del_t+self.tau[j])
            ind = ind_low & ind_high
            if np.any(ind):
                grad[0, j, ind] = Model._delCBF1(self.M0[ind], self.alpha[ind],
                                                 self.lambd[ind], x[0, ind],
                                                 f_sc,
                                                 self.T1[ind], x[1, ind],
                                                 del_t_sc, self.T1b[ind], t[j],
                                                 self.tau[j, ind],
                                                 T1prinv[ind],
                                                 expAtt[ind])
                grad[1, j, ind] = Model._delATT1(self.M0[ind], self.alpha[ind],
                                                 self.lambd[ind], x[0, ind],
                                                 f_sc,
                                                 self.T1[ind], x[1, ind],
                                                 del_t_sc, self.T1b[ind], t[j],
                                                 self.tau[j, ind],
                                                 T1prinv[ind],
                                                 expAtt[ind])

            ind = self.t[j] >= del_t + self.tau[j]
            if np.any(ind):
                grad[0, j, ind] = Model._delCBF2(self.M0[ind], self.alpha[ind],
                                                 self.lambd[ind], x[0, ind],
                                                 f_sc,
                                                 self.T1[ind], x[1, ind],
                                                 del_t_sc, self.T1b[ind], t[j],
                                                 self.tau[j, ind],
                                                 T1prinv[ind],
                                                 expAtt[ind])
                grad[1, j, ind] = Model._delATT2(self.M0[ind], self.alpha[ind],
                                                 self.lambd[ind], x[0, ind],
                                                 f_sc,
                                                 self.T1[ind], x[1, ind],
                                                 del_t_sc, self.T1b[ind], t[j],
                                                 self.tau[j, ind],
                                                 T1prinv[ind],
                                                 expAtt[ind])
        grad[~np.isfinite(grad)] = 1e-20
        grad = np.array(grad, dtype=DTYPE)
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        images = self._execute_forward_3D(x)
        f = np.abs(x[0, ...] * self.uk_scale[0] / self.dscale)
        del_t = np.abs(x[1, ...] * self.uk_scale[1])*60
#        del_t[f <= 15] = 0
        f_min = f.min()
        f_max = f.max()
        del_t_min = del_t.min()
        del_t_max = del_t.max()
        ind = 80  # int(images.shape[-1]/2) 30, 60
        if dim_2D:
            if not self.figure:
                plt.ion()
                self.figure, self.ax = plt.subplots(1, 2, figsize=(12, 5))
                self.f_plot = self.ax[0].imshow((f))
                self.ax[0].set_title('Proton Density in a.u.')
                self.ax[0].axis('off')
                self.figure.colorbar(self.f_plot, ax=self.ax[0])
                self.del_t_plot = self.ax[1].imshow((del_t))
                self.ax[1].set_title('del_t in  ms')
                self.ax[1].axis('off')
                self.figure.colorbar(self.del_t_plot, ax=self.ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self.f_plot.set_data((f))
                self.f_plot.set_clim([f_min, f_max])
                self.del_t_plot.set_data((del_t))
                self.del_t_plot.set_clim([del_t_min, del_t_max])
                plt.draw()
                plt.pause(1e-10)
        else:
            [z, y, x] = f.shape
            self.ax = []
            if not self.figure:
                plt.ion()
                self.figure = plt.figure(figsize=(12, 6))
                self.figure.subplots_adjust(hspace=0, wspace=0)
                self.gs = gridspec.GridSpec(
                    3, 6,
                    width_ratios=[
                        x / (20 * z), x / z, 1, x / z, 1, x / (20 * z)],
                    height_ratios=[x / z, 1, x/z])
                self.figure.tight_layout()
                self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs:
                    self.ax.append(plt.subplot(grid))
                    self.ax[-1].axis('off')

                self.f_plot = self.ax[1].imshow(
                    (f[int(self.NSlice / 2), ...]))
                self.f_plot_cor = self.ax[7].imshow(
                    (f[:, int(f.shape[1] / 2), ...]))
                self.f_plot_sag = self.ax[2].imshow(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.ax[1].set_title('CBF', color='white')
                self.ax[1].set_anchor('SE')
                self.ax[2].set_anchor('SW')
                self.ax[7].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 0])
                cbar = self.figure.colorbar(self.f_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)

                self.del_t_plot = self.ax[3].imshow(
                    (del_t[int(self.NSlice / 2), ...]))
                self.del_t_plot_cor = self.ax[9].imshow(
                    (del_t[:, int(del_t.shape[1] / 2), ...]))
                self.del_t_plot_sag = self.ax[4].imshow(
                    np.flip((del_t[:, :, int(del_t.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ATT', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[9].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 5])
                cbar = self.figure.colorbar(self.del_t_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')
                plt.draw()
                plt.pause(1e-10)
                self.plot_ax = plt.subplot(self.gs[-1, :])

                self.time_course_ref = self.plot_ax.scatter(
                    self.t*60, np.real(
                        self.images[:, int(self.NSlice/2), ind, ind]),
                    color='g', marker="2")
                self.time_course = self.plot_ax.plot(
                    self.t*60, np.real(
                        images[:, int(self.NSlice/2), ind, ind]), 'r')[0]
                self.plot_ax.set_ylim(
                    np.real(self.images[:,
                                        int(self.NSlice/2),
                                        ind,
                                        ind]).min() - np.real(
                            self.images[:,
                                        int(self.NSlice/2),
                                        ind,
                                        ind]).min() * 0.01,
                    np.real(self.images[:,
                                        int(self.NSlice/2),
                                        ind,
                                        ind]).max() + np.real(
                           self.images[:,
                                       int(self.NSlice/2),
                                       ind,
                                       ind]).max() * 0.01)
                for spine in self.plot_ax.spines:
                    self.plot_ax.spines[spine].set_color('white')
                plt.draw()
                plt.show()
                plt.pause(1e-4)
            else:
                self.f_plot.set_data((f[int(self.NSlice / 2), ...]))
                self.f_plot_cor.set_data((f[:, int(f.shape[1] / 2), ...]))
                self.f_plot_sag.set_data(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.f_plot.set_clim([f_min, f_max])
                self.f_plot_cor.set_clim([f_min, f_max])
                self.f_plot_sag.set_clim([f_min, f_max])
                self.del_t_plot.set_data((del_t[int(self.NSlice / 2), ...]))
                self.del_t_plot_cor.set_data(
                    (del_t[:, int(del_t.shape[1] / 2), ...]))
                self.del_t_plot_sag.set_data(
                    np.flip((del_t[:, :, int(del_t.shape[-1] / 2)]).T, 1))
                self.del_t_plot.set_clim([del_t_min, del_t_max])
                self.del_t_plot_sag.set_clim([del_t_min, del_t_max])
                self.del_t_plot_cor.set_clim([del_t_min, del_t_max])

                self.time_course.set_ydata(
                    np.real(images[:, int(self.NSlice/2), ind, ind]))
                self.plot_ax.set_ylim(
                    np.real(self.images[:,
                                        int(self.NSlice/2),
                                        ind,
                                        ind]).min() - np.real(
                            self.images[:,
                                        int(self.NSlice/2),
                                        ind,
                                        ind]).min() * 0.01,
                    np.real(self.images[:,
                                        int(self.NSlice/2),
                                        ind,
                                        ind]).max() + np.real(
                           self.images[:,
                                       int(self.NSlice/2),
                                       ind,
                                       ind]).max() * 0.01)
                plt.draw()
                plt.pause(1e-10)

    def _set_init_scales(self, images):
        test_f = 10 * self.dscale * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        test_del_t = 1/60 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        x = np.array([test_f,
                      test_del_t], dtype=DTYPE)
        return x
