#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyqmri.models.template import BaseModel, constraints, DTYPE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne
plt.ion()


def _S(M0, M0sc, ADC, meanADC, kurt, b):
    return ne.evaluate("M0 * M0sc * \
                       exp(1 / 6 * meanADC**2 * b**2 * kurt - ADC * b)")


def _diffexp(ADC, meanADC, kurt, b):
    return ne.evaluate("exp(1 / 6 * meanADC**2 * b**2 * kurt - ADC * b)")


def _bKurtmeanADC(b, meanADC, kurt):
    return ne.evaluate("b*meanADC*kurt")


def _comADCGrad(M0, gradM0, b):
    return ne.evaluate("2*M0*gradM0*b")


def _comKurtfac(meanADC, comADCGrad, b):
    return ne.evaluate("1/12*meanADC**2*comADCGrad*b")


def _gradKurtDia(Kurtscale, bdir, comKurtfac):
    return ne.evaluate("comKurtfac * Kurtscale * bdir**4")


def _gradKurtOff2(comKurtfac, Kurtscale, bdir1, bdir2):
    return ne.evaluate("comKurtfac * Kurtscale * 4 *  bdir1**3 * bdir2")


def _gradKurtOffSym(comKurtfac, Kurtscale, bdir1, bdir2):
    return ne.evaluate("comKurtfac * Kurtscale * 6 *  bdir1**2 * bdir2**2")


def _gradKurtOff3(comKurtfac, Kurtscale, bdir1, bdir2, bdir3):
    return ne.evaluate("comKurtfac * Kurtscale * 12 * \
                       bdir1**2 * bdir2 * bdir3")


class Model(BaseModel):
    """ Test """
    def __init__(self, par):
        super().__init__(par)

        self.b = np.ones((self.NScan, 1, 1, 1))
        self.dir = par["DWI_dir"].T

        for i in range(self.NScan):
            self.b[i, ...] = par["b_value"][i] * np.ones((1, 1, 1))

        if np.max(self.b) > 100:
            self.b /= 1000

        self.dir = self.dir[:, None, None, None, :]
        self.dirsqr = self.dir**2
        self.dircub = self.dir**3
        self.dir4th = self.dir**4

        par["unknowns_TGV"] = 22
        par["unknowns_H1"] = 0
        par["unknowns"] = par["unknowns_TGV"] + par["unknowns_H1"]
        self.uk_scale = []
        for j in range(par["unknowns_TGV"] + par["unknowns_H1"]):
            self.uk_scale.append(1)

        self.unknowns = par["unknowns_TGV"] + par["unknowns_H1"]
        try:
            self.b0 = np.flip(
                np.transpose(par["file"]["b0"][()], (0, 2, 1)), 0)
        except KeyError:
            print("No b0 image provided")
            self.b0 = None

        self.constraints.append(
            constraints(
                0 / self.uk_scale[0],
                100 / self.uk_scale[0],
                False))
        for j in range(3):
            self.constraints.append(
                constraints(
                    (-4 / self.uk_scale[2*j+1]),
                    (4 / self.uk_scale[2*j+1]),
                    True))
            self.constraints.append(
                constraints(
                    (-3e0 / self.uk_scale[2*j+2]),
                    (3e0 / self.uk_scale[2*j+2]),
                    True))

        Kmax = 2
        for j in range(3):
            self.constraints.append(
                constraints(
                    (0 / self.uk_scale[j+7]),
                    (Kmax / self.uk_scale[j+7]),
                    True))

        for j in range(6):
            self.constraints.append(constraints(
                (-Kmax / self.uk_scale[j + 10]),
                (Kmax / self.uk_scale[j + 10]), True))

        for j in range(3):
            self.constraints.append(constraints(
                (0 / self.uk_scale[j + 16]),
                (Kmax / self.uk_scale[j + 16]), True))

        for j in range(3):
            self.constraints.append(constraints(
                (-Kmax / self.uk_scale[j + 19]),
                (Kmax / self.uk_scale[j + 19]), True))

    def rescale(self, x):
        out = np.zeros_like(x)
        out[0] = x[0, ...] * self.uk_scale[0]
        out[1] = (np.real(x[1, ...]**2) * self.uk_scale[1]**2)
        out[2] = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[1, ...] * self.uk_scale[1]))
        out[3] = (np.real(x[2, ...]**2 * self.uk_scale[2]**2 +
                          x[3, ...]**2 * self.uk_scale[3]**2))
        out[4] = (np.real(x[4, ...] * self.uk_scale[4] *
                          x[1, ...] * self.uk_scale[1]))
        out[5] = (np.real(x[4, ...]**2 * self.uk_scale[4]**2 +
                          x[5, ...]**2 * self.uk_scale[5]**2 +
                          x[6, ...]**2 * self.uk_scale[6]**2))
        out[6] = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[4, ...] * self.uk_scale[4] +
                          x[6, ...] * self.uk_scale[6] *
                          x[3, ...] * self.uk_scale[3]))

        for j in np.arange(7, 22):
            out[j] = x[j]*self.uk_scale[j]
        return out

    def _execute_forward_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_gradient_2D(self, x, islice):
        print("2D Functions not implemented")
        raise NotImplementedError

    def _execute_forward_3D(self, x):
        ADC = x[1, ...]**2 * self.uk_scale[1]**2 * self.dirsqr[..., 0] + \
              (x[2, ...]**2 * self.uk_scale[2]**2 +
               x[3, ...]**2 * self.uk_scale[3]**2) * self.dirsqr[..., 1] + \
              (x[4, ...]**2 * self.uk_scale[4]**2 +
               x[5, ...]**2 * self.uk_scale[5]**2 +
               x[6, ...]**2 * self.uk_scale[6]**2) * self.dirsqr[..., 2] +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[1, ...] * self.uk_scale[1]) * \
              self.dir[..., 0] * self.dir[..., 1] + \
              2 * (x[4, ...] * self.uk_scale[4] *
                   x[1, ...] * self.uk_scale[1]) *\
              self.dir[..., 0] * self.dir[..., 2] +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[4, ...] * self.uk_scale[4] +
                   x[6, ...] * self.uk_scale[6] *
                   x[3, ...] * self.uk_scale[3]) * \
              self.dir[..., 1] * self.dir[..., 2]

        meanADC = 1 / 3 * (x[1, ...]**2 * self.uk_scale[1]**2 +
                           (x[2, ...]**2 * self.uk_scale[2]**2 +
                            x[3, ...]**2 * self.uk_scale[3]**2) +
                           (x[4, ...]**2 * self.uk_scale[4]**2 +
                            x[5, ...]**2 * self.uk_scale[5]**2 +
                            x[6, ...]**2 * self.uk_scale[6]**2))

        kurt = (
          x[7, ...] * self.uk_scale[7] * self.dir4th[..., 0] +
          x[8, ...] * self.uk_scale[8] * self.dir4th[..., 1] +
          x[9, ...] * self.uk_scale[9] * self.dir4th[..., 2] +
          4 * x[10, ...] * self.uk_scale[10] *
          self.dircub[..., 0] * self.dir[..., 1] +
          4 * x[11, ...] * self.uk_scale[11] *
          self.dircub[..., 0] * self.dir[..., 2] +
          4 * x[12, ...] * self.uk_scale[12] *
          self.dircub[..., 1] * self.dir[..., 0] +
          4 * x[13, ...] * self.uk_scale[13] *
          self.dircub[..., 1] * self.dir[..., 2] +
          4 * x[14, ...] * self.uk_scale[14] *
          self.dircub[..., 2] * self.dir[..., 0] +
          4 * x[15, ...] * self.uk_scale[15] *
          self.dircub[..., 2] * self.dir[..., 1] +
          6 * x[16, ...] * self.uk_scale[16] *
          self.dirsqr[..., 0] * self.dirsqr[..., 1] +
          6 * x[17, ...] * self.uk_scale[17] *
          self.dirsqr[..., 0] * self.dirsqr[..., 2] +
          6 * x[18, ...] * self.uk_scale[18] *
          self.dirsqr[..., 1] * self.dirsqr[..., 2] +
          12 * x[19, ...] * self.uk_scale[19] *
          self.dirsqr[..., 0] * self.dir[..., 1] * self.dir[..., 2] +
          12 * x[20, ...] * self.uk_scale[20] *
          self.dir[..., 0] * self.dirsqr[..., 1] * self.dir[..., 2] +
          12 * x[21, ...] * self.uk_scale[21] *
          self.dir[..., 0] * self.dir[..., 1] * self.dirsqr[..., 2])

        S = _S(x[0], self.uk_scale[0],
               ADC, meanADC, kurt, self.b).astype(DTYPE)

        S *= self.phase
        S[~np.isfinite(S)] = 0
#        S[:, meanADC**2*np.max(self.b)**2 > 32] = 0
        return S

    def _execute_gradient_3D(self, x):
        ADC = x[1, ...]**2 * self.uk_scale[1]**2 * self.dirsqr[..., 0] + \
              (x[2, ...]**2 * self.uk_scale[2]**2 +
               x[3, ...]**2 * self.uk_scale[3]**2) * self.dirsqr[..., 1] + \
              (x[4, ...]**2 * self.uk_scale[4]**2 +
               x[5, ...]**2 * self.uk_scale[5]**2 +
               x[6, ...]**2 * self.uk_scale[6]**2) * self.dirsqr[..., 2] +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[1, ...] * self.uk_scale[1]) * \
              self.dir[..., 0] * self.dir[..., 1] + \
              2 * (x[4, ...] * self.uk_scale[4] *
                   x[1, ...] * self.uk_scale[1]) *\
              self.dir[..., 0] * self.dir[..., 2] +\
              2 * (x[2, ...] * self.uk_scale[2] *
                   x[4, ...] * self.uk_scale[4] +
                   x[6, ...] * self.uk_scale[6] *
                   x[3, ...] * self.uk_scale[3]) * \
              self.dir[..., 1] * self.dir[..., 2]

        meanADC = 1 / 3 * (x[1, ...]**2 * self.uk_scale[1]**2 +
                           (x[2, ...]**2 * self.uk_scale[2]**2 +
                            x[3, ...]**2 * self.uk_scale[3]**2) +
                           (x[4, ...]**2 * self.uk_scale[4]**2 +
                            x[5, ...]**2 * self.uk_scale[5]**2 +
                            x[6, ...]**2 * self.uk_scale[6]**2))

        kurt = (
          x[7, ...] * self.uk_scale[7] * self.dir4th[..., 0] +
          x[8, ...] * self.uk_scale[8] * self.dir4th[..., 1] +
          x[9, ...] * self.uk_scale[9] * self.dir4th[..., 2] +
          4 * x[10, ...] * self.uk_scale[10] *
          self.dircub[..., 0] * self.dir[..., 1] +
          4 * x[11, ...] * self.uk_scale[11] *
          self.dircub[..., 0] * self.dir[..., 2] +
          4 * x[12, ...] * self.uk_scale[12] *
          self.dircub[..., 1] * self.dir[..., 0] +
          4 * x[13, ...] * self.uk_scale[13] *
          self.dircub[..., 1] * self.dir[..., 2] +
          4 * x[14, ...] * self.uk_scale[14] *
          self.dircub[..., 2] * self.dir[..., 0] +
          4 * x[15, ...] * self.uk_scale[15] *
          self.dircub[..., 2] * self.dir[..., 1] +
          6 * x[16, ...] * self.uk_scale[16] *
          self.dirsqr[..., 0] * self.dirsqr[..., 1] +
          6 * x[17, ...] * self.uk_scale[17] *
          self.dirsqr[..., 0] * self.dirsqr[..., 2] +
          6 * x[18, ...] * self.uk_scale[18] *
          self.dirsqr[..., 1] * self.dirsqr[..., 2] +
          12 * x[19, ...] * self.uk_scale[19] *
          self.dirsqr[..., 0] * self.dir[..., 1] * self.dir[..., 2] +
          12 * x[20, ...] * self.uk_scale[20] *
          self.dir[..., 0] * self.dirsqr[..., 1] * self.dir[..., 2] +
          12 * x[21, ...] * self.uk_scale[21] *
          self.dir[..., 0] * self.dir[..., 1] * self.dirsqr[..., 2])

        grad_M0 = self.uk_scale[0] * _diffexp(ADC, meanADC, kurt, self.b) * \
            self.phase

        bKurtmeanADC = _bKurtmeanADC(self.b, meanADC, kurt)
        comADCGrad = _comADCGrad(x[0, ...], grad_M0, self.b)

        grad_ADC_x = x[0, ...]*grad_M0*(2/9*x[1, ...]*self.uk_scale[1]**2*self.b**2*meanADC*kurt - self.b*(2*x[1, ...]*self.uk_scale[1]**2*self.dirsqr[..., 0] + 2*self.uk_scale[1]*x[2, ...]*self.uk_scale[2]*self.dir[..., 0]*self.dir[..., 1] + 2*self.uk_scale[1]*x[4, ...]*self.uk_scale[4]*self.dir[..., 0]*self.dir[..., 2]))

        grad_ADC_xy = x[0, ...]*grad_M0*(2/9*x[2, ...]*self.uk_scale[2]**2*self.b**2*meanADC*kurt - self.b*(2*x[1, ...]*self.uk_scale[1]*self.uk_scale[2]*self.dir[..., 0]*self.dir[..., 1] + 2*x[2, ...]*self.uk_scale[2]**2*self.dirsqr[..., 1] + 2*self.uk_scale[2]*x[4, ...]*self.uk_scale[4]*self.dir[..., 1]*self.dir[..., 2]))

        grad_ADC_y = x[0, ...]*grad_M0*(2/9*x[3, ...]*self.uk_scale[3]**2*self.b**2*meanADC*kurt - self.b*(2*x[3, ...]*self.uk_scale[3]**2*self.dirsqr[..., 1] + 2*self.uk_scale[3]*x[6, ...]*self.uk_scale[6]*self.dir[..., 1]*self.dir[..., 2]))

        grad_ADC_xz = x[0, ...]*grad_M0*(2/9*x[4, ...]*self.uk_scale[4]**2*self.b**2*meanADC*kurt - self.b*(2*x[1, ...]*self.uk_scale[1]*self.uk_scale[4]*self.dir[..., 0]*self.dir[..., 2] + 2*x[2, ...]*self.uk_scale[2]*self.uk_scale[4]*self.dir[..., 1]*self.dir[..., 2] + 2*x[4, ...]*self.uk_scale[4]**2*self.dirsqr[..., 2]))

        grad_ADC_z = x[0, ...]*grad_M0*(2/9*x[5, ...]*self.uk_scale[5]**2*self.b**2*meanADC*kurt - 2*x[5, ...]*self.uk_scale[5]**2*self.b*self.dirsqr[..., 2])

        grad_ADC_yz = x[0, ...]*grad_M0*(2/9*x[6, ...]*self.uk_scale[6]**2*self.b**2*meanADC*kurt - self.b*(2*x[3, ...]*self.uk_scale[3]*self.uk_scale[6]*self.dir[..., 1]*self.dir[..., 2] + 2*x[6, ...]*self.uk_scale[6]**2*self.dirsqr[..., 2]))

        del bKurtmeanADC

        comKurtfac = _comKurtfac(meanADC, comADCGrad, self.b)

        del comADCGrad

        grad_kurt_xxxx = _gradKurtDia(self.uk_scale[7],
                                      self.dir[..., 0], comKurtfac)
        grad_kurt_yyyy = _gradKurtDia(self.uk_scale[8],
                                      self.dir[..., 0], comKurtfac)
        grad_kurt_zzzz = _gradKurtDia(self.uk_scale[9],
                                      self.dir[..., 0], comKurtfac)

        grad_kurt_xxxy = _gradKurtOff2(comKurtfac,
                                       self.uk_scale[10],
                                       self.dir[..., 0],
                                       self.dir[..., 1])
        grad_kurt_xxxz = _gradKurtOff2(comKurtfac,
                                       self.uk_scale[11],
                                       self.dir[..., 0],
                                       self.dir[..., 2])
        grad_kurt_xyyy = _gradKurtOff2(comKurtfac,
                                       self.uk_scale[12],
                                       self.dir[..., 1],
                                       self.dir[..., 0])
        grad_kurt_yyyz = _gradKurtOff2(comKurtfac,
                                       self.uk_scale[13],
                                       self.dir[..., 1],
                                       self.dir[..., 2])
        grad_kurt_xzzz = _gradKurtOff2(comKurtfac,
                                       self.uk_scale[14],
                                       self.dir[..., 2],
                                       self.dir[..., 0])
        grad_kurt_yzzz = _gradKurtOff2(comKurtfac,
                                       self.uk_scale[15],
                                       self.dir[..., 2],
                                       self.dir[..., 1])

        grad_kurt_xxyy = _gradKurtOffSym(comKurtfac,
                                         self.uk_scale[16],
                                         self.dir[..., 0],
                                         self.dir[..., 1])
        grad_kurt_xxzz = _gradKurtOffSym(comKurtfac,
                                         self.uk_scale[17],
                                         self.dir[..., 0],
                                         self.dir[..., 2])
        grad_kurt_yyzz = _gradKurtOffSym(comKurtfac,
                                         self.uk_scale[18],
                                         self.dir[..., 1],
                                         self.dir[..., 2])

        grad_kurt_xxyz = _gradKurtOff3(comKurtfac,
                                       self.uk_scale[19],
                                       self.dir[..., 0],
                                       self.dir[..., 1],
                                       self.dir[..., 2])
        grad_kurt_xyyz = _gradKurtOff3(comKurtfac,
                                       self.uk_scale[20],
                                       self.dir[..., 1],
                                       self.dir[..., 0],
                                       self.dir[..., 2])
        grad_kurt_xyzz = _gradKurtOff3(comKurtfac,
                                       self.uk_scale[21],
                                       self.dir[..., 2],
                                       self.dir[..., 0],
                                       self.dir[..., 1])
        del comKurtfac

        grad = np.array([grad_M0,
                         grad_ADC_x,
                         grad_ADC_xy,
                         grad_ADC_y,
                         grad_ADC_xz,
                         grad_ADC_z,
                         grad_ADC_yz,
                         grad_kurt_xxxx,
                         grad_kurt_yyyy,
                         grad_kurt_zzzz,
                         grad_kurt_xxxy,
                         grad_kurt_xxxz,
                         grad_kurt_xyyy,
                         grad_kurt_yyyz,
                         grad_kurt_xzzz,
                         grad_kurt_yzzz,
                         grad_kurt_xxyy,
                         grad_kurt_xxzz,
                         grad_kurt_yyzz,
                         grad_kurt_xxyz,
                         grad_kurt_xyyz,
                         grad_kurt_xyzz],
                        dtype=DTYPE)

        grad[~np.isfinite(grad)] = 0
#        grad[:, :, meanADC**2*np.max(self.b)**2 > 32] = 0
        return grad

    def plot_unknowns(self, x, dim_2D=False):
        M0 = np.abs(x[0, ...]) * self.uk_scale[0]
        ADC_x = (np.real(x[1, ...]**2 * self.uk_scale[1]**2))
        ADC_xy = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[1, ...] * self.uk_scale[1]))
        M0_min = M0.min()
        M0_max = M0.max()
        ADC_x_min = ADC_x.min()
        ADC_x_max = ADC_x.max()
        ADC_xy_min = ADC_xy.min()
        ADC_xy_max = ADC_xy.max()

        ADC_y = (np.real(x[2, ...]**2 * self.uk_scale[2]**2 +
                         x[3, ...]**2 * self.uk_scale[3]**2))
        ADC_xz = (np.real(x[4, ...] * self.uk_scale[4] *
                          x[1, ...] * self.uk_scale[1]))
        ADC_y_min = ADC_y.min()
        ADC_y_max = ADC_y.max()
        ADC_xz_min = ADC_xz.min()
        ADC_xz_max = ADC_xz.max()

        ADC_z = (np.real(x[4, ...]**2 * self.uk_scale[4]**2 +
                         x[5, ...]**2 * self.uk_scale[5]**2 +
                         x[6, ...]**2 * self.uk_scale[6]**2))
        ADC_yz = (np.real(x[2, ...] * self.uk_scale[2] *
                          x[4, ...] * self.uk_scale[4] +
                          x[6, ...] * self.uk_scale[6] *
                          x[3, ...] * self.uk_scale[3]))
        ADC_z_min = ADC_z.min()
        ADC_z_max = ADC_z.max()
        ADC_yz_min = ADC_yz.min()
        ADC_yz_max = ADC_yz.max()

        kurt_xxxx = np.real(x[7, ...] * self.uk_scale[7])
        kurt_xxxx_min = kurt_xxxx.min()
        kurt_xxxx_max = kurt_xxxx.max()

        kurt_yyyy = np.real(x[8, ...] * self.uk_scale[8])
        kurt_yyyy_min = kurt_yyyy.min()
        kurt_yyyy_max = kurt_yyyy.max()

        kurt_zzzz = np.real(x[9, ...] * self.uk_scale[9])
        kurt_zzzz_min = kurt_zzzz.min()
        kurt_zzzz_max = kurt_zzzz.max()

        kurt = []
        kurt_min = []
        kurt_max = []

        for j in range(12):
            tmp = np.real(np.real(x[10 + j, ...] * self.uk_scale[10 + j]))
            tmp1 = np.concatenate(
                (tmp[int(self.NSlice / 2), ...],
                 tmp[:, int(tmp.shape[1] / 2), ...].T), -1)
            tmp2 = np.concatenate(
                (np.flip(
                    (tmp[:, :, int(tmp.shape[-1] / 2)]), 1),
                 np.zeros((tmp.shape[0], tmp.shape[0]))), -1)
            kurt.append(np.concatenate((tmp1, tmp2), 0))
            kurt_min.append(kurt[j].min())
            kurt_max.append(kurt[j].max())

        if dim_2D:
            if not self.figure:
                plt.ion()
                self.figure, self.ax = plt.subplots(1, 2, figsize=(12, 5))
                self.M0_plot = self.ax[0].imshow((M0))
                self.ax[0].set_title('Proton Density in a.u.')
                self.ax[0].axis('off')
                self.figure.colorbar(self.M0_plot, ax=self.ax[0])
                self.ADC_x_plot = self.ax[1].imshow((ADC_x))
                self.ax[1].set_title('ADC_x in  ms')
                self.ax[1].axis('off')
                self.figure.colorbar(self.ADC_x_plot, ax=self.ax[1])
                self.figure.tight_layout()
                plt.draw()
                plt.pause(1e-10)
            else:
                self.M0_plot.set_data((M0))
                self.M0_plot.set_clim([M0_min, M0_max])
                self.ADC_x_plot.set_data((ADC_x))
                self.ADC_x_plot.set_clim([ADC_x_min, ADC_x_max])
                plt.draw()
                plt.pause(1e-10)
        else:
            [z, y, x] = M0.shape
            self.ax = []
            self.ax_phase = []
            self.ax_kurt = []
            if not self.figure:
                plt.ion()
                self.figure = plt.figure(figsize=(12, 6))
                self.figure.subplots_adjust(hspace=0, wspace=0)
                self.gs = gridspec.GridSpec(8,
                                            10,
                                            width_ratios=[
                                                x / (20 * z),
                                                x / z,
                                                1,
                                                x / z,
                                                1,
                                                x / (20 * z),
                                                x / (2 * z),
                                                x / z,
                                                1,
                                                x / (20 * z)],
                                            height_ratios=[
                                                x / z,
                                                1,
                                                x / z,
                                                1,
                                                x / z,
                                                1,
                                                x / z,
                                                1])
                self.figure.tight_layout()
                self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs:
                    self.ax.append(plt.subplot(grid))
                    self.ax[-1].axis('off')

                self.M0_plot = self.ax[1].imshow(
                    (M0[int(self.NSlice / 2), ...]))
                self.M0_plot_cor = self.ax[11].imshow(
                    (M0[:, int(M0.shape[1] / 2), ...]))
                self.M0_plot_sag = self.ax[2].imshow(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self.ax[1].set_title('Proton Density in a.u.', color='white')
                self.ax[1].set_anchor('SE')
                self.ax[2].set_anchor('SW')
                self.ax[11].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 0])
                cbar = self.figure.colorbar(self.M0_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_x_plot = self.ax[3].imshow(
                    (ADC_x[int(self.NSlice / 2), ...]))
                self.ADC_x_plot_cor = self.ax[13].imshow(
                    (ADC_x[:, int(ADC_x.shape[1] / 2), ...]))
                self.ADC_x_plot_sag = self.ax[4].imshow(
                    np.flip((ADC_x[:, :, int(ADC_x.shape[-1] / 2)]).T, 1))
                self.ax[3].set_title('ADC_x', color='white')
                self.ax[3].set_anchor('SE')
                self.ax[4].set_anchor('SW')
                self.ax[13].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 5])
                cbar = self.figure.colorbar(self.ADC_x_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_xy_plot = self.ax[7].imshow(
                    (ADC_xy[int(self.NSlice / 2), ...]))
                self.ADC_xy_plot_cor = self.ax[17].imshow(
                    (ADC_xy[:, int(ADC_x.shape[1] / 2), ...]))
                self.ADC_xy_plot_sag = self.ax[8].imshow(
                    np.flip((ADC_xy[:, :, int(ADC_x.shape[-1] / 2)]).T, 1))
                self.ax[7].set_title('ADC_xy', color='white')
                self.ax[7].set_anchor('SE')
                self.ax[8].set_anchor('SW')
                self.ax[17].set_anchor('NE')
                cax = plt.subplot(self.gs[:2, 9])
                cbar = self.figure.colorbar(self.ADC_xy_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_y_plot = self.ax[23].imshow(
                    (ADC_y[int(self.NSlice / 2), ...]))
                self.ADC_y_plot_cor = self.ax[33].imshow(
                    (ADC_y[:, int(ADC_y.shape[1] / 2), ...]))
                self.ADC_y_plot_sag = self.ax[24].imshow(
                    np.flip((ADC_y[:, :, int(ADC_y.shape[-1] / 2)]).T, 1))
                self.ax[23].set_title('ADC_y', color='white')
                self.ax[23].set_anchor('SE')
                self.ax[24].set_anchor('SW')
                self.ax[33].set_anchor('NE')
                cax = plt.subplot(self.gs[2:4, 5])
                cbar = self.figure.colorbar(self.ADC_y_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_xz_plot = self.ax[27].imshow(
                    (ADC_xz[int(self.NSlice / 2), ...]))
                self.ADC_xz_plot_cor = self.ax[37].imshow(
                    (ADC_xz[:, int(ADC_y.shape[1] / 2), ...]))
                self.ADC_xz_plot_sag = self.ax[28].imshow(
                    np.flip((ADC_xz[:, :, int(ADC_y.shape[-1] / 2)]).T, 1))
                self.ax[27].set_title('ADC_xz', color='white')
                self.ax[27].set_anchor('SE')
                self.ax[28].set_anchor('SW')
                self.ax[37].set_anchor('NE')
                cax = plt.subplot(self.gs[2:4, 9])
                cbar = self.figure.colorbar(self.ADC_xz_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_z_plot = self.ax[43].imshow(
                    (ADC_z[int(self.NSlice / 2), ...]))
                self.ADC_z_plot_cor = self.ax[53].imshow(
                    (ADC_z[:, int(ADC_z.shape[1] / 2), ...]))
                self.ADC_z_plot_sag = self.ax[44].imshow(
                    np.flip((ADC_z[:, :, int(ADC_z.shape[-1] / 2)]).T, 1))
                self.ax[43].set_title('ADC_z', color='white')
                self.ax[43].set_anchor('SE')
                self.ax[44].set_anchor('SW')
                self.ax[53].set_anchor('NE')
                cax = plt.subplot(self.gs[4:6, 5])
                cbar = self.figure.colorbar(self.ADC_z_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.ADC_yz_plot = self.ax[47].imshow(
                    (ADC_yz[int(self.NSlice / 2), ...]))
                self.ADC_yz_plot_cor = self.ax[57].imshow(
                    (ADC_yz[:, int(ADC_z.shape[1] / 2), ...]))
                self.ADC_yz_plot_sag = self.ax[48].imshow(
                    np.flip((ADC_yz[:, :, int(ADC_z.shape[-1] / 2)]).T, 1))
                self.ax[47].set_title('ADC_yz', color='white')
                self.ax[47].set_anchor('SE')
                self.ax[48].set_anchor('SW')
                self.ax[57].set_anchor('NE')
                cax = plt.subplot(self.gs[4:6, 9])
                cbar = self.figure.colorbar(self.ADC_yz_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.kurt_xxxx_plot = self.ax[61].imshow(
                    (kurt_xxxx[int(self.NSlice / 2), ...]))
                self.kurt_xxxx_plot_cor = self.ax[71].imshow(
                    (kurt_xxxx[:, int(kurt_xxxx.shape[1] / 2), ...]))
                self.kurt_xxxx_plot_sag = self.ax[62].imshow(
                    np.flip(
                        (kurt_xxxx[:, :, int(kurt_xxxx.shape[-1] / 2)]).T, 1))
                self.ax[61].set_title('kurtosis_x in a.u.', color='white')
                self.ax[61].set_anchor('SE')
                self.ax[62].set_anchor('SW')
                self.ax[71].set_anchor('NE')
                cax = plt.subplot(self.gs[6:8, 0])
                cbar = self.figure.colorbar(self.kurt_xxxx_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                cax.yaxis.set_ticks_position('left')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.kurt_yyyy_plot = self.ax[63].imshow(
                    (kurt_yyyy[int(self.NSlice / 2), ...]))
                self.kurt_yyyy_plot_cor = self.ax[73].imshow(
                    (kurt_yyyy[:, int(kurt_yyyy.shape[1] / 2), ...]))
                self.kurt_yyyy_plot_sag = self.ax[64].imshow(
                    np.flip(
                        (kurt_yyyy[:, :, int(kurt_yyyy.shape[-1] / 2)]).T, 1))
                self.ax[63].set_title('kurtosis_y in a.u.', color='white')
                self.ax[63].set_anchor('SE')
                self.ax[64].set_anchor('SW')
                self.ax[73].set_anchor('NE')
                cax = plt.subplot(self.gs[6:8, 5])
                cbar = self.figure.colorbar(self.kurt_yyyy_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                self.kurt_zzzz_plot = self.ax[67].imshow(
                    (kurt_zzzz[int(self.NSlice / 2), ...]))
                self.kurt_zzzz_plot_cor = self.ax[77].imshow(
                    (kurt_zzzz[:, int(kurt_zzzz.shape[1] / 2), ...]))
                self.kurt_zzzz_plot_sag = self.ax[68].imshow(
                    np.flip(
                        (kurt_zzzz[:, :, int(kurt_zzzz.shape[-1] / 2)]).T, 1))
                self.ax[67].set_title('kurtosis_z in a.u.', color='white')
                self.ax[67].set_anchor('SE')
                self.ax[68].set_anchor('SW')
                self.ax[77].set_anchor('NE')
                cax = plt.subplot(self.gs[6:8, 9])
                cbar = self.figure.colorbar(self.kurt_zzzz_plot, cax=cax)
                cbar.ax.tick_params(labelsize=12, colors='white')
                for spine in cbar.ax.spines:
                    cbar.ax.spines[spine].set_color('white')

                plt.draw()
                plt.pause(1e-10)

                self.figure.canvas.draw_idle()
                plot_dim = int(np.ceil(np.sqrt(len(kurt))))
                plt.ion()
                self.figure_kurt = plt.figure(figsize=(12, 6))
                self.figure_kurt.subplots_adjust(hspace=0.3, wspace=0)
                wd_ratio = np.tile([1, 1 / 20, 1 / (5)], plot_dim)
                self.gs_kurt = gridspec.GridSpec(
                    plot_dim, 3 * plot_dim,
                    width_ratios=wd_ratio, hspace=0.3, wspace=0)
                self.figure_kurt.tight_layout()
                self.figure_kurt.patch.set_facecolor(plt.cm.viridis.colors[0])
                for grid in self.gs_kurt:
                    self.ax_kurt.append(plt.subplot(grid))
                    self.ax_kurt[-1].axis('off')
                self.kurt_plot = []
                for j in range(len(kurt)):
                    self.kurt_plot.append(
                        self.ax_kurt[3 * j].imshow((kurt[j])))
                    self.ax_kurt[3 *
                                 j].set_title('Kurtosis of dir: ' +
                                              str(j), color='white')
                    self.ax_kurt[3 * j + 1].axis('on')
                    cbar = self.figure.colorbar(
                        self.kurt_plot[j], cax=self.ax_kurt[3 * j + 1])
                    cbar.ax.tick_params(labelsize=12, colors='white')
                    for spine in cbar.ax.spines:
                        cbar.ax.spines[spine].set_color('white')

                plt.draw()
                plt.pause(1e-10)

            else:
                self.M0_plot.set_data((M0[int(self.NSlice / 2), ...]))
                self.M0_plot_cor.set_data((M0[:, int(M0.shape[1] / 2), ...]))
                self.M0_plot_sag.set_data(
                    np.flip((M0[:, :, int(M0.shape[-1] / 2)]).T, 1))
                self.M0_plot.set_clim([M0_min, M0_max])
                self.M0_plot_cor.set_clim([M0_min, M0_max])
                self.M0_plot_sag.set_clim([M0_min, M0_max])

                self.ADC_x_plot.set_data((ADC_x[int(self.NSlice / 2), ...]))
                self.ADC_x_plot_cor.set_data(
                    (ADC_x[:, int(ADC_x.shape[1] / 2), ...]))
                self.ADC_x_plot_sag.set_data(
                    np.flip((ADC_x[:, :, int(ADC_x.shape[-1] / 2)]).T, 1))
                self.ADC_x_plot.set_clim([ADC_x_min, ADC_x_max])
                self.ADC_x_plot_sag.set_clim([ADC_x_min, ADC_x_max])
                self.ADC_x_plot_cor.set_clim([ADC_x_min, ADC_x_max])

                self.ADC_xy_plot.set_data((ADC_xy[int(self.NSlice / 2), ...]))
                self.ADC_xy_plot_cor.set_data(
                    (ADC_xy[:, int(ADC_xy.shape[1] / 2), ...]))
                self.ADC_xy_plot_sag.set_data(
                    np.flip((ADC_xy[:, :, int(ADC_xy.shape[-1] / 2)]).T, 1))
                self.ADC_xy_plot.set_clim([ADC_xy_min, ADC_xy_max])
                self.ADC_xy_plot_sag.set_clim([ADC_xy_min, ADC_xy_max])
                self.ADC_xy_plot_cor.set_clim([ADC_xy_min, ADC_xy_max])

                self.ADC_y_plot.set_data((ADC_y[int(self.NSlice / 2), ...]))
                self.ADC_y_plot_cor.set_data(
                    (ADC_y[:, int(ADC_y.shape[1] / 2), ...]))
                self.ADC_y_plot_sag.set_data(
                    np.flip((ADC_y[:, :, int(ADC_y.shape[-1] / 2)]).T, 1))
                self.ADC_y_plot.set_clim([ADC_y_min, ADC_y_max])
                self.ADC_y_plot_sag.set_clim([ADC_y_min, ADC_y_max])
                self.ADC_y_plot_cor.set_clim([ADC_y_min, ADC_y_max])

                self.ADC_xz_plot.set_data((ADC_xz[int(self.NSlice / 2), ...]))
                self.ADC_xz_plot_cor.set_data(
                    (ADC_xz[:, int(ADC_xz.shape[1] / 2), ...]))
                self.ADC_xz_plot_sag.set_data(
                    np.flip((ADC_xz[:, :, int(ADC_xz.shape[-1] / 2)]).T, 1))
                self.ADC_xz_plot.set_clim([ADC_xz_min, ADC_xz_max])
                self.ADC_xz_plot_sag.set_clim([ADC_xz_min, ADC_xz_max])
                self.ADC_xz_plot_cor.set_clim([ADC_xz_min, ADC_xz_max])

                self.ADC_z_plot.set_data((ADC_z[int(self.NSlice / 2), ...]))
                self.ADC_z_plot_cor.set_data(
                    (ADC_z[:, int(ADC_z.shape[1] / 2), ...]))
                self.ADC_z_plot_sag.set_data(
                    np.flip((ADC_z[:, :, int(ADC_z.shape[-1] / 2)]).T, 1))
                self.ADC_z_plot.set_clim([ADC_z_min, ADC_z_max])
                self.ADC_z_plot_sag.set_clim([ADC_z_min, ADC_z_max])
                self.ADC_z_plot_cor.set_clim([ADC_z_min, ADC_z_max])

                self.ADC_yz_plot.set_data((ADC_yz[int(self.NSlice / 2), ...]))
                self.ADC_yz_plot_cor.set_data(
                    (ADC_yz[:, int(ADC_yz.shape[1] / 2), ...]))
                self.ADC_yz_plot_sag.set_data(
                    np.flip((ADC_yz[:, :, int(ADC_yz.shape[-1] / 2)]).T, 1))
                self.ADC_yz_plot.set_clim([ADC_yz_min, ADC_yz_max])
                self.ADC_yz_plot_sag.set_clim([ADC_yz_min, ADC_yz_max])
                self.ADC_yz_plot_cor.set_clim([ADC_yz_min, ADC_yz_max])

                self.kurt_xxxx_plot.set_data(
                    (kurt_xxxx[int(self.NSlice / 2), ...]))
                self.kurt_xxxx_plot_cor.set_data(
                    (kurt_xxxx[:, int(kurt_xxxx.shape[1] / 2), ...]))
                self.kurt_xxxx_plot_sag.set_data(
                    np.flip(
                        (kurt_xxxx[:, :, int(kurt_xxxx.shape[-1] / 2)]).T, 1))
                self.kurt_xxxx_plot.set_clim([kurt_xxxx_min, kurt_xxxx_max])
                self.kurt_xxxx_plot_cor.set_clim(
                    [kurt_xxxx_min, kurt_xxxx_max])
                self.kurt_xxxx_plot_sag.set_clim(
                    [kurt_xxxx_min, kurt_xxxx_max])

                self.kurt_yyyy_plot.set_data(
                    (kurt_yyyy[int(self.NSlice / 2), ...]))
                self.kurt_yyyy_plot_cor.set_data(
                    (kurt_yyyy[:, int(kurt_yyyy.shape[1] / 2), ...]))
                self.kurt_yyyy_plot_sag.set_data(
                    np.flip(
                        (kurt_yyyy[:, :, int(kurt_yyyy.shape[-1] / 2)]).T, 1))
                self.kurt_yyyy_plot.set_clim([kurt_yyyy_min, kurt_yyyy_max])
                self.kurt_yyyy_plot_cor.set_clim(
                    [kurt_yyyy_min, kurt_yyyy_max])
                self.kurt_yyyy_plot_sag.set_clim(
                    [kurt_yyyy_min, kurt_yyyy_max])

                self.kurt_zzzz_plot.set_data(
                    (kurt_zzzz[int(self.NSlice / 2), ...]))
                self.kurt_zzzz_plot_cor.set_data(
                    (kurt_zzzz[:, int(kurt_zzzz.shape[1] / 2), ...]))
                self.kurt_zzzz_plot_sag.set_data(
                    np.flip(
                        (kurt_zzzz[:, :, int(kurt_zzzz.shape[-1] / 2)]).T, 1))
                self.kurt_zzzz_plot.set_clim([kurt_zzzz_min, kurt_zzzz_max])
                self.kurt_zzzz_plot_cor.set_clim(
                    [kurt_zzzz_min, kurt_zzzz_max])
                self.kurt_zzzz_plot_sag.set_clim(
                    [kurt_zzzz_min, kurt_zzzz_max])

                self.figure.canvas.draw_idle()

                for j in range(len(kurt)):
                    self.kurt_plot[j].set_data((kurt[j]))
                    self.kurt_plot[j].set_clim([kurt_min[j], kurt_max[j]])

                self.figure_kurt.canvas.draw_idle()

                self.figure.canvas.draw_idle()

                for j in range(len(kurt)):
                    self.kurt_plot[j].set_data((kurt[j]))
                    self.kurt_plot[j].set_clim([kurt_min[j], kurt_max[j]])

                self.figure_kurt.canvas.draw_idle()

                plt.draw()
                plt.pause(1e-10)

    def computeInitialGuess(self, *args):
        self.phase = np.exp(1j*(np.angle(args[0])-np.angle(args[0][0])))
        if self.b0 is not None:
            test_M0 = self.b0
        else:
            test_M0 = args[0][0]

        ADC_x = 1 * np.ones(
            (self.NSlice, self.dimY, self.dimX), dtype=DTYPE)
        kurt_yyyy = 1 * np.ones((self.NSlice, self.dimY,
                                 self.dimX), dtype=DTYPE)
        kurt_xxxx = 0 * np.ones((self.NSlice, self.dimY,
                                 self.dimX), dtype=DTYPE)

        self.guess = np.array(
                [
                    test_M0,
                    ADC_x,
                    0 * ADC_x,
                    ADC_x,
                    0 * ADC_x,
                    ADC_x,
                    0 * ADC_x,
                    kurt_yyyy,
                    kurt_yyyy,
                    kurt_yyyy,
                    kurt_xxxx,
                    kurt_xxxx,
                    kurt_xxxx,
                    kurt_xxxx,
                    kurt_xxxx,
                    kurt_xxxx,
                    kurt_yyyy/2,
                    kurt_yyyy/2,
                    kurt_yyyy/2,
                    kurt_xxxx,
                    kurt_xxxx,
                    kurt_xxxx],
                dtype=DTYPE)
