#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints
import numexpr as ne
plt.ion()


def _expAttT1b(del_t, del_t_sc, T1b):
    return ne.evaluate(
        "exp(-(del_t*del_t_sc)/T1b)")


def _T1pr(T1, f, f_sc, lambd):
    return ne.evaluate(
        "1/T1+f*f_sc/lambd")


def _S1(M0, alpha, lambd, f, T1, T1p, del_t,  t, tau, expAttT1b):
    return ne.evaluate(
        "2*alpha*M0/lambd * f/T1p * expAttT1b * \
        (1-exp(-(t-del_t) * T1p))")


def _S2(M0, alpha, lambd, f, T1, T1p, del_t, t, tau, expAttT1b):
    return ne.evaluate(
        "2*alpha*M0/lambd * f/T1p * expAttT1b * \
         exp(-(t-del_t-tau) * T1p) * \
         (1-exp(-tau * T1p))")


def _delCBF1(M0, alpha, lambd, f, f_sc, T1, del_t,
             del_t_sc, T1b, t, tau, T1p, expAttT1b):
    return ne.evaluate(
        "(-2*M0*f*f_sc**2*(del_t*del_t_sc - t) *\
         exp((del_t*del_t_sc - t) * T1p) * expAttT1b / (lambd**2*T1p) -\
         2*M0*f*f_sc**2 * (-exp((del_t*del_t_sc - t) * T1p) + 1) * \
         expAttT1b / (lambd**2*T1p**2) +\
         2*M0*f_sc*(-exp((del_t*del_t_sc - t) * T1p) + 1) * expAttT1b /\
         (lambd*T1p))*alpha")


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


def _delATT1(M0, alpha, lambd, f, f_sc, T1, del_t,
             del_t_sc, T1b, t, tau, T1p, expAttT1b):
    return ne.evaluate(
        "(-2*M0*del_t_sc*f*f_sc * exp((del_t*del_t_sc - t) * T1p) *\
          expAttT1b/lambd -\
          2*M0*del_t_sc*f*f_sc*(- exp((del_t*del_t_sc - t) * T1p) + 1) *\
          expAttT1b / (T1b*lambd*T1p))*alpha")


def _delATT2(M0, alpha, lambd, f, f_sc, T1, del_t,
             del_t_sc, T1b, t, tau, T1p, expAttT1b):
    return ne.evaluate(
        "(2*M0*del_t_sc*f*f_sc * (1 - exp(-tau*T1p)) *\
          exp(T1p * (del_t*del_t_sc - t + tau)) * expAttT1b/lambd -\
          2*M0*del_t_sc*f*f_sc * (1 - exp(-tau*T1p)) *\
          exp(T1p * (del_t*del_t_sc - t + tau)) * expAttT1b /\
          (T1b*lambd*T1p))*alpha")


class Model(BaseModel):
    def __init__(self, par):
        super().__init__(par)
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

        par["unknowns_TGV"] = 2
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"]+par["unknowns_H1"]
        self.unknowns = par["unknowns"]

        for j in range(self.unknowns):
            self.uk_scale.append(1)

        self.constraints.append(
            constraints(0,
                        200,
                        False))
        self.constraints.append(
            constraints(0.01/60,
                        5/60,
                        True))
        self._ind1 = 35
        self._ind2 = 46

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
                     dtype=self._DTYPE)

        T1prinv = _T1pr(self.T1, x[0], self.uk_scale[0], self.lambd)
        expAtt = _expAttT1b(x[1], self.uk_scale[1], self.T1b)
        for j in range(self.t.shape[0]):
            ind_low = self.t[j] >= del_t
            ind_high = self.t[j] < (del_t+self.tau[j])
            ind = ind_low & ind_high
            if np.any(ind):
                S[j, ind] = _S1(self.M0[ind], self.alpha[ind],
                                self.lambd[ind], f[ind], self.T1[ind],
                                T1prinv[ind], del_t[ind],
                                self.t[j], self.tau[j, ind], expAtt[ind])
            ind = self.t[j] >= del_t + self.tau[j]
            if np.any(ind):
                S[j, ind] = _S2(self.M0[ind], self.alpha[ind],
                                self.lambd[ind], f[ind], self.T1[ind],
                                T1prinv[ind], del_t[ind],
                                self.t[j], self.tau[j, ind], expAtt[ind])
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=self._DTYPE)
        return S

    def _execute_gradient_3D(self, x):
        f_sc = self.uk_scale[0]
        del_t_sc = self.uk_scale[1]
        del_t = x[1]*del_t_sc
        grad = np.zeros((self.unknowns, self.NScan,
                         self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        t = self.t
        T1prinv = _T1pr(self.T1, x[0], self.uk_scale[0], self.lambd)
        expAtt = _expAttT1b(x[1], self.uk_scale[1], self.T1b)
        for j in range(self.t.shape[0]):
            ind_low = self.t[j] >= del_t
            ind_high = self.t[j] < (del_t+self.tau[j])
            ind = ind_low & ind_high
            if np.any(ind):
                grad[0, j, ind] = _delCBF1(self.M0[ind], self.alpha[ind],
                                           self.lambd[ind], x[0, ind],
                                           f_sc,
                                           self.T1[ind], x[1, ind],
                                           del_t_sc, self.T1b[ind], t[j],
                                           self.tau[j, ind],
                                           T1prinv[ind],
                                           expAtt[ind])
                grad[1, j, ind] = _delATT1(self.M0[ind], self.alpha[ind],
                                           self.lambd[ind], x[0, ind],
                                           f_sc,
                                           self.T1[ind], x[1, ind],
                                           del_t_sc, self.T1b[ind], t[j],
                                           self.tau[j, ind],
                                           T1prinv[ind],
                                           expAtt[ind])

            ind = self.t[j] >= del_t + self.tau[j]
            if np.any(ind):
                grad[0, j, ind] = _delCBF2(self.M0[ind], self.alpha[ind],
                                           self.lambd[ind], x[0, ind],
                                           f_sc,
                                           self.T1[ind], x[1, ind],
                                           del_t_sc, self.T1b[ind], t[j],
                                           self.tau[j, ind],
                                           T1prinv[ind],
                                           expAtt[ind])
                grad[1, j, ind] = _delATT2(self.M0[ind], self.alpha[ind],
                                           self.lambd[ind], x[0, ind],
                                           f_sc,
                                           self.T1[ind], x[1, ind],
                                           del_t_sc, self.T1b[ind], t[j],
                                           self.tau[j, ind],
                                           T1prinv[ind],
                                           expAtt[ind])
        grad[~np.isfinite(grad)] = 1e-20
        grad = np.array(grad, dtype=self._DTYPE)
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        unknowns = self.rescale(x)
        tmp_x = unknowns["data"]
        
        images = self._execute_forward_3D(x) / self.dscale
        
        f = np.abs(tmp_x[0, ...]/ self.dscale)
        
        del_t = np.abs(tmp_x[1, ...])*60
#        del_t[f <= 15] = 0
        f_min = f.min()
        f_max = f.max()
        del_t_min = del_t.min()
        del_t_max = del_t.max()
        # ind1 = 46
        # ind2 = 35# int(images.shape[-1]/2) 30, 60
        off = 0
        [z, y, x] = f.shape
        if len(self.t.shape) == 4:
            t = self.t[:, int(self.NSlice/2), 0, 0]
        else:
            t = self.t
        if dim_2D:
            pass
        else:
            if not self._figure:
                self.ax = []
                plt.ion()
                self._figure = plt.figure(figsize=(12, 6))
                self._figure.subplots_adjust(hspace=0, wspace=0)
                self.gs = gridspec.GridSpec(
                    4, 6,
                    width_ratios=[
                        x / (20 * z), x / z, 1, x / z, 1, x / (20 * z)],
                    height_ratios=[x / z, 1, x / z, x / z])
                self._figure.tight_layout()
                self._figure.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs:
                    self.ax.append(plt.subplot(grid))
                    self.ax[-1].axis('off')

                self.ax[1].volume = f
                self.ax[1].index = int(self.NSlice / 2)
                self.f_plot = self.ax[1].imshow(
                    (f[int(self.NSlice / 2), ...]))
                self.ax[7].volume = np.swapaxes(f, 0, 1)
                self.ax[7].index = int(f.shape[1] / 2)
                self.f_plot_cor = self.ax[7].imshow(
                    (f[:, int(f.shape[1] / 2), ...]))
                self.ax[2].volume = f.T
                self.ax[2].index = int(f.shape[-1] / 2)
                self.f_plot_sag = self.ax[2].imshow(
                    np.flip((f[:, :, int(f.shape[-1] / 2)]).T, 1))
                self.ax[1].set_title('CBF', color='white')
                self.ax[1].set_anchor('SE')
                self.ax[2].set_anchor('SW')
                self.ax[7].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 0])
                cbar = self._figure.colorbar(self.f_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ax[3].volume = del_t
                self.ax[3].index = int(self.NSlice / 2)
                self.del_t_plot = self.ax[3].imshow(
                    (del_t[int(self.NSlice / 2)+off, ...]))
                self.ax[9].volume = np.swapaxes(del_t, 0, 1)
                self.ax[9].index = int(del_t.shape[1] / 2)
                self.del_t_plot_cor = self.ax[9].imshow(
                    (del_t[:, int(del_t.shape[1] / 2), ...]))
                self.ax[4].volume = del_t.T
                self.ax[4].index = int(del_t.shape[-1] / 2)
                self.del_t_plot_sag = self.ax[4].imshow(
                    np.flip((del_t[:, :, int(del_t.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ATT', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[9].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 5])
                cbar = self._figure.colorbar(self.del_t_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                plt.draw()
                plt.pause(1e-10)

                self.plot_ax = plt.subplot(self.gs[-1, :])
                self.plot_ax.set_title("Time course", color='w')
                self.time_course_ref = []

                self.time_course_ref.append(self.plot_ax.plot(
                    t*60, np.real(
                        self.images[...,
                                    int(self.NSlice/2),
                                    self._ind2, self._ind1]),
                    'x')[0])
                
                self.plot_ax.set_prop_cycle(None)

                self.time_course = self.plot_ax.plot(
                    t*60, np.real(
                        images[..., int(self.NSlice/2),
                               self._ind2, self._ind1]))
                
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
                plt.pause(1e-10)
                
                self._figure.canvas.mpl_connect(
                    'button_press_event',
                    self.onclick)
                self._figure.canvas.mpl_connect(
                    'scroll_event',
                    self.onscroll)
                
            else:
                self.ax[1].volume = f
                self.ax[7].volume = np.swapaxes(f, 0, 1)
                self.ax[2].volume = f.T

                self.ax[3].volume = del_t
                self.ax[9].volume = np.swapaxes(del_t, 0, 1)
                self.ax[4].volume = del_t.T
                
                self.ax[1].images[0].set_array(self.ax[1].volume[self.ax[1].index])
                self.ax[2].images[0].set_array(self.ax[2].volume[self.ax[2].index])
                self.ax[7].images[0].set_array(self.ax[7].volume[self.ax[7].index])
                
                self.f_plot.set_clim([f_min, f_max])
                self.f_plot_cor.set_clim([f_min, f_max])
                self.f_plot_sag.set_clim([f_min, f_max])
                
                self.ax[3].images[0].set_array(self.ax[3].volume[self.ax[3].index])
                self.ax[4].images[0].set_array(self.ax[4].volume[self.ax[4].index])
                self.ax[9].images[0].set_array(self.ax[9].volume[self.ax[9].index])
                
                self.del_t_plot.set_clim([del_t_min, del_t_max])
                self.del_t_plot_sag.set_clim([del_t_min, del_t_max])
                self.del_t_plot_cor.set_clim([del_t_min, del_t_max])

                self.time_course[0].set_ydata(
                    np.real(images[:, self.ax[1].index, 
                                   self._ind2, self._ind1]))
                
                self.plot_ax.set_ylim(
                    np.minimum(np.real(images[...,
                                              self.ax[1].index,
                                              self._ind2,
                                              self._ind1]).min(),
                               np.real(self.images[...,
                                                   self.ax[1].index,
                                                   self._ind2,
                                                   self._ind1]).min()),
                    1.2*np.maximum(np.real(images[...,
                                                  self.ax[1].index,
                                                  self._ind2,
                                                  self._ind1]).max(),
                                   np.real(self.images[...,
                                                       self.ax[1].index,
                                                       self._ind2,
                                                       self._ind1]).max()))
                plt.draw()
                plt.pause(1e-10)
                
    def onclick(self, event):
        if event.inaxes in [self.ax[1], self.ax[3]]:
            self._ind1 = int(event.xdata)
            self._ind2 = int(event.ydata)

            self.time_course_ref[0].set_ydata(np.real(
                    self.images[...,
                                self.ax[1].index,
                                self._ind2, self._ind1]))
            self.plot_ax.set_ylim(
                (np.real(self.images[...,
                                     self.ax[1].index,
                                     self._ind2,
                                     self._ind1]).min()),
                1.2*(np.real(self.images[...,
                                         self.ax[1].index,
                                         self._ind2,
                                         self._ind1]).max()))
            
    def onscroll(self, event):
        if event.inaxes in [self.ax[1], self.ax[3]]:
            fig = event.canvas.figure
            ax = [self.ax[1], self.ax[3]]
        
        elif event.inaxes in [self.ax[2], self.ax[4]]:
            fig = event.canvas.figure
            ax = [self.ax[2], self.ax[4]]
                        
        elif event.inaxes in [self.ax[7], self.ax[9]]:
            fig = event.canvas.figure
            ax = [self.ax[7], self.ax[9]]
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
                
                        
                        

    def computeInitialGuess(self, *args):
        self.dscale = args[1]
        self.constraints[0].update(1/self.dscale)
        self.images = args[0]/args[1]
        test_f = 30 * self.dscale * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        test_del_t = 1/60 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=self._DTYPE)
        x = np.array([test_f,
                      test_del_t], dtype=self._DTYPE)
        self.guess = x
