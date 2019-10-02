#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqmri.models.template import BaseModel, constraints, DTYPE

plt.ion()
unknowns_TGV = 1
unknowns_H1 = 0


class Model(BaseModel):
    def __init__(self, par, images):
        super().__init__(par)
        self.constraints = []
        self.figure_img = []
        self.dscale = par["dscale"]

        for j in range(unknowns_TGV + unknowns_H1):
            self.uk_scale.append(1)
        self.guess = self._set_init_scales(images)

        for j in range(unknowns_TGV + unknowns_H1):
            self.constraints.append(
                constraints(-100 / self.uk_scale[j],
                            100 / self.uk_scale[j],
                            False))

    def rescale(self, x):
        tmp_x = np.copy(x)
        for j in range(x.shape[0]):
            tmp_x[j] *= self.uk_scale[j]
        return tmp_x

    def _execute_forward_2D(self, x, islice):
        S = np.zeros_like(x)
        for j in range(S.shape[0]):
            S = x[j, ...] * self.uk_scale[j]
        S[~np.isfinite(S)] = 1e-20
        S = np.array(S, dtype=DTYPE)
        return S

    def _execute_gradient_2D(self, x, islice):
        grad_M0 = self.uk_scale[0] * np.ones_like(x)
        grad = np.array([grad_M0], dtype=DTYPE)
        grad[~np.isfinite(grad)] = 1e-20
        return grad

    def _execute_forward_3D(self, x):
        S = np.zeros_like(x)
        for j in range(S.shape[0]):
            S[j, ...] = x[j, ...] * self.uk_scale[j]
        S[~np.isfinite(S)] = 1e-20
        return S

    def _execute_gradient_3D(self, x):
        grad_M0 = np.zeros(((unknowns_TGV+unknowns_H1, )+x.shape), dtype=DTYPE)
        for j in range(x.shape[0]):
            grad_M0[j, ...] = self.uk_scale[j]*np.ones_like(x)
        grad_M0[~np.isfinite(grad_M0)] = 1e-20
        return grad_M0

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.zeros_like(x)
        M0_min = []
        M0_max = []
        M0 = np.abs(M0)
        for j in range(x.shape[0]):
            M0[j, ...] = np.abs(x[j, ...] * self.uk_scale[j])
            M0_min.append(M0[j].min())
            M0_max.append(M0[j].max())
#        M0 = np.transpose(M0, [0, 2, 1, 3])
        if dim_2D:
            raise NotImplementedError("2D Not Implemented")
        else:
            [z, y, x] = M0.shape[1:]
            if not self.figure_img:
                plt.ion()
                self.ax_img = []
                plot_dim = int(np.ceil(np.sqrt(M0.shape[0])))
                self.figure_img = plt.figure(figsize=(12, 6))
                self.figure_img.subplots_adjust(hspace=0.3, wspace=0)
                wd_ratio = np.tile([1, 1 / 20, 1 / (5)], plot_dim)
                self.gs_kurt = gridspec.GridSpec(
                    plot_dim, 3 * plot_dim,
                    width_ratios=wd_ratio, hspace=0.3, wspace=0)
                self.figure_img.tight_layout()
                self.figure_img.patch.set_facecolor('black')
                for grid in self.gs_kurt:
                    self.ax_img.append(plt.subplot(grid))
                    self.ax_img[-1].axis('off')
                self.image_plot = []
                for j in range(M0.shape[0]):
                    self.image_plot.append(
                        self.ax_img[3 * j].imshow(
                            (M0[j, int(z/2)]),
                            vmin=M0_min[j],
                            vmax=M0_max[j], cmap='gray'))
                    self.ax_img[3 *
                                j].set_title('Image: ' +
                                             str(j), color='white')
                    self.ax_img[3 * j + 1].axis('on')
                    cbar = self.figure_img.colorbar(
                        self.image_plot[j], cax=self.ax_img[3 * j + 1])
                    cbar.ax.tick_params(labelsize=12, colors='white')
                    for spine in cbar.ax.spines:
                        cbar.ax.spines[spine].set_color('white')
                    plt.draw()
                    plt.pause(1e-10)
            else:
                for j in range(M0.shape[0]):
                    self.image_plot[j].set_data((M0[j, int(z/2)]))
                    self.image_plot[j].set_clim([M0_min[j],
                                                 M0_max[j]])

                self.figure_img.canvas.draw_idle()
                plt.draw()
                plt.pause(1e-10)

    def _set_init_scales(self, images):
        return np.abs(images).astype(DTYPE)
