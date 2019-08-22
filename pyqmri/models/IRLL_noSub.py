import numpy as np
import matplotlib.pyplot as plt
plt.ion()

DTYPE = np.complex64


class constraint:
    def __init__(self, min_val=-np.inf, max_val=np.inf, real_const=False):
        self.min = min_val
        self.max = max_val
        self.real = real_const


class IRLL_Model:

    def __init__(self, par, images):

        self.constraints = []
        self.NSlice = par["NSlice"]
        self.TR = par['time_per_slice'] - \
            (par['tau'] * par['Nproj_measured'] + par['gradient_delay'])
        self.fa = par["flip_angle(s)"] * np.pi / 180
        self.fa_corr = par["fa_corr"]

        self.uk_scale.append(1)
        self.uk_scale.append(3000)

        self.Nproj_measured = par["Nproj_measured"]
        self.tau = par["tau"]
        self.td = par["gradient_delay"]
        self.NLL = par["NScan"]
        self.Nproj = par["Nproj"]
        self.dimY = par["dimY"]
        self.dimX = par["dimX"]

        phi_corr = np.zeros_like(par["fa_corr"], dtype=DTYPE)
        phi_corr = np.real(
            self.fa) * np.real(
                par["fa_corr"]) + 1j * np.imag(
                    self.fa) * np.imag(par["fa_corr"])

        self.sin_phi = np.sin(phi_corr)
        self.cos_phi = np.cos(phi_corr)

        test_T1 = np.reshape(
            np.linspace(
                10,
                5500,
                self.dimX *
                self.dimY *
                self.NSlice),
            (self.NSlice,
             self.dimY,
             self.dimX))
        test_M0 = 1
        G_x = self.execute_forward_3D(
            np.array(
                [
                    test_M0 /
                    self.uk_scale[0] *
                    np.ones(
                        (self.NSlice,
                         self.dimY,
                         self.dimX),
                        dtype=DTYPE),
                    1 /
                    self.uk_scale[1] *
                    test_T1 *
                    np.ones(
                        (self.NSlice,
                         self.dimY,
                         self.dimX),
                        dtype=DTYPE)],
                dtype=DTYPE))
        self.uk_scale[0] = self.uk_scale[0] * \
            np.median(np.abs(images)) / np.median(np.abs(G_x))

        DG_x = self.execute_gradient_3D(
            np.array(
                [
                    test_M0 /
                    self.uk_scale[0] *
                    np.ones(
                        (self.NSlice,
                         self.dimY,
                         self.dimX),
                        dtype=DTYPE),
                    1 /
                    self.uk_scale[1] *
                    test_T1 *
                    np.ones(
                        (self.NSlice,
                         self.dimY,
                         self.dimX),
                        dtype=DTYPE)],
                dtype=DTYPE))
        self.uk_scale[1] = self.uk_scale[1] * np.linalg.norm(
            np.abs(DG_x[0, ...])) / np.linalg.norm(np.abs(DG_x[1, ...]))

        self.uk_scale[1] = self.uk_scale[1]
        DG_x = self.execute_gradient_3D(
            np.array(
                [
                    test_M0 /
                    self.uk_scale[0] *
                    np.ones(
                        (self.NSlice,
                         self.dimY,
                         self.dimX),
                        dtype=DTYPE),
                    1 /
                    self.uk_scale[1] *
                    test_T1 *
                    np.ones(
                        (self.NSlice,
                         self.dimY,
                         self.dimX),
                        dtype=DTYPE)],
                dtype=DTYPE))

        self.guess = np.array([1 /
                               self.uk_scale[0] *
                               np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE), 800 /
                               self.uk_scale[1] *
                               np.ones((self.NSlice, self.dimY, self.dimX), dtype=DTYPE)])

        self.constraints.append(constraint(-300, 300, False))
        self.constraints.append(
            constraint(
                10 / self.T1_sc,
                5500 / self.T1_sc,
                True))

    def execute_forward_2D(self, x, islice):
        S = np.zeros((self.NLL, self.Nproj, self.dimY, self.dimX), dtype=DTYPE)
        M0_sc = self.M0_sc
        T1_sc = self.T1_sc
        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self.sin_phi[islice, ...]
        cos_phi = self.cos_phi[islice, ...]
        N = self.Nproj_measured
        T1 = x[1, ...]
        M0 = x[0, ...]
        Etr = np.exp(-TR / (T1 * T1_sc))
        Etd = np.exp(-td / (T1 * T1_sc))
        Etau = np.exp(-tau / (T1 * T1_sc))
        cosEtau = cos_phi * Etau
        cosEtauN = cosEtau**(N - 1)
        F = (1 - Etau) / (-cosEtau + 1)
        Q = (-cos_phi * F * (-cosEtauN + 1) * Etr * Etd + 1 - 2 *
             Etd + Etr * Etd) / (cos_phi * cosEtauN * Etr * Etd + 1)
        Q_F = Q - F
        for i in range(self.NLL):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1
                S[i, j, ...] = M0 * M0_sc * sin_phi * \
                    ((cosEtau)**(n - 1) * (Q_F) + F)

        return np.mean(S, axis=1)

    def execute_gradient_2D(self, x, islice):
        grad = np.zeros(
            (2,
             self.NLL,
             self.Nproj,
             self.dimY,
             self.dimX),
            dtype=DTYPE)
        M0_sc = self.M0_sc
        T1_sc = self.T1_sc
        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self.sin_phi[islice, ...]
        cos_phi = self.cos_phi[islice, ...]
        N = self.Nproj_measured
        T1 = x[1, ...]
        M0 = x[0, ...]
        # Precompute
        Etr = np.exp(-TR / (T1 * T1_sc))
        Etd = np.exp(-td / (T1 * T1_sc))
        Etau = np.exp(-tau / (T1 * T1_sc))
        cosEtau = cos_phi * Etau
        cosEtauN = cosEtau**(N - 1)
        F = (1 - Etau) / (-cosEtau + 1)
        Q = (-cos_phi * F * (-cosEtauN + 1) * Etr * Etd + 1 - 2 *
             Etd + Etr * Etd) / (cos_phi * cosEtauN * Etr * Etd + 1)
        Q_F = Q - F
        tmp1 = ((TR * Etr * Etd / (T1**2 * T1_sc) - TR * (1 - Etau) *
                 (-(Etau * cos_phi)**(N - 1) + 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)) + tau * (Etau * cos_phi)**(N - 1) *
                 (1 - Etau) * (N - 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)) + tau * (-(Etau * cos_phi)**(N - 1) + 1) *
                 Etr * Etau * Etd * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)) - tau * (1 - Etau) * (-(Etau * cos_phi)**(N - 1) + 1)
                 * Etr * Etau * Etd * cos_phi**2 / (T1**2 * T1_sc * (1 - Etau * cos_phi)**2) - 2 * td * Etd / (T1**2 * T1_sc) + td * Etr * Etd / (T1**2 * T1_sc)
                 - td * (1 - Etau) * (-(Etau * cos_phi)**(N - 1) + 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi))) /
                ((Etau * cos_phi)**(N - 1) * Etr * Etd * cos_phi + 1) + (-TR * (Etau * cos_phi)**(N - 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc) -
                                                                         tau * (Etau * cos_phi)**(N - 1) * (N - 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc) - td * (Etau * cos_phi)**(N - 1) * Etr * Etd * cos_phi /
                                                                         (T1**2 * T1_sc)) * (1 - 2 * Etd + Etr * Etd - (1 - Etau) * (-(Etau * cos_phi)**(N - 1) + 1) * Etr * Etd * cos_phi / (1 - Etau * cos_phi)) /
                ((Etau * cos_phi)**(N - 1) * Etr * Etd * cos_phi + 1)**2 + tau * Etau / (T1**2 * T1_sc * (1 - Etau * cos_phi)) -
                tau * (1 - Etau) * Etau * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)**2))
        tmp2 = ((1 - 2 * Etd + Etr * Etd - (1 - Etau) * (-(Etau * cos_phi)**(N - 1) + 1) * Etr * Etd * cos_phi / (1 - Etau * cos_phi)
                 ) / ((Etau * cos_phi) ** (N - 1) * Etr * Etd * cos_phi + 1) - (1 - Etau) / (1 - Etau * cos_phi)) / (T1**2 * T1_sc)
        tmp3 = - tau * Etau / (T1**2 * T1_sc * (1 - Etau * cos_phi)) + tau * (
            1 - Etau) * Etau * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)**2)

        for i in range(self.NLL):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1

                grad[0, i, j, ...] = M0_sc * sin_phi * \
                    ((cosEtau)**(n - 1) * (Q_F) + F)

                grad[1, i, j, ...] = M0 * M0_sc * ((Etau * cos_phi)**(n - 1) * tmp1 + tau * (
                    Etau * cos_phi)**(n - 1) * (n - 1) * tmp2 + tmp3) * sin_phi

        return np.mean(grad, axis=2)

    def execute_forward_3D(self, x):
        S = np.zeros(
            (self.NLL,
             self.Nproj,
             self.NSlice,
             self.dimY,
             self.dimX),
            dtype=DTYPE)
        M0_sc = self.M0_sc
        T1_sc = self.T1_sc
        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self.sin_phi
        cos_phi = self.cos_phi
        N = self.Nproj_measured
        T1 = x[1, ...]
        M0 = x[0, ...]
        Etr = np.exp(-TR / (T1 * T1_sc))
        Etd = np.exp(-td / (T1 * T1_sc))
        Etau = np.exp(-tau / (T1 * T1_sc))
        cosEtau = cos_phi * Etau
        cosEtauN = cosEtau**(N - 1)
        F = (1 - Etau) / (-cosEtau + 1)
        Q = (-cos_phi * F * (-cosEtauN + 1) * Etr * Etd + 1 - 2 *
             Etd + Etr * Etd) / (cos_phi * cosEtauN * Etr * Etd + 1)
        Q_F = Q - F
        for i in range(self.NLL):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1
                S[i, j, ...] = M0 * M0_sc * sin_phi * \
                    ((cosEtau)**(n - 1) * (Q_F) + F)

        return np.mean(S, axis=1)

    def execute_gradient_3D(self, x):
        grad = np.zeros(
            (2,
             self.NLL,
             self.Nproj,
             self.NSlice,
             self.dimY,
             self.dimX),
            dtype=DTYPE)
        M0_sc = self.M0_sc
        T1_sc = self.T1_sc
        TR = self.TR
        tau = self.tau
        td = self.td
        sin_phi = self.sin_phi
        cos_phi = self.cos_phi
        N = self.Nproj_measured
        T1 = x[1, ...]
        M0 = x[0, ...]
        # Precompute
        Etr = np.exp(-TR / (T1 * T1_sc))
        Etd = np.exp(-td / (T1 * T1_sc))
        Etau = np.exp(-tau / (T1 * T1_sc))
        cosEtau = cos_phi * Etau
        cosEtauN = cosEtau**(N - 1)
        F = (1 - Etau) / (-cosEtau + 1)
        Q = (-cos_phi * F * (-cosEtauN + 1) * Etr * Etd + 1 - 2 *
             Etd + Etr * Etd) / (cos_phi * cosEtauN * Etr * Etd + 1)
        Q_F = Q - F
        tmp1 = ((TR * Etr * Etd / (T1**2 * T1_sc) - TR * (1 - Etau) *
                 (-(Etau * cos_phi)**(N - 1) + 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)) + tau * (Etau * cos_phi)**(N - 1) *
                 (1 - Etau) * (N - 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)) + tau * (-(Etau * cos_phi)**(N - 1) + 1) *
                 Etr * Etau * Etd * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)) - tau * (1 - Etau) * (-(Etau * cos_phi)**(N - 1) + 1)
                 * Etr * Etau * Etd * cos_phi**2 / (T1**2 * T1_sc * (1 - Etau * cos_phi)**2) - 2 * td * Etd / (T1**2 * T1_sc) + td * Etr * Etd / (T1**2 * T1_sc)
                 - td * (1 - Etau) * (-(Etau * cos_phi)**(N - 1) + 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi))) /
                ((Etau * cos_phi)**(N - 1) * Etr * Etd * cos_phi + 1) + (-TR * (Etau * cos_phi)**(N - 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc) -
                                                                         tau * (Etau * cos_phi)**(N - 1) * (N - 1) * Etr * Etd * cos_phi / (T1**2 * T1_sc) - td * (Etau * cos_phi)**(N - 1) * Etr * Etd * cos_phi /
                                                                         (T1**2 * T1_sc)) * (1 - 2 * Etd + Etr * Etd - (1 - Etau) * (-(Etau * cos_phi)**(N - 1) + 1) * Etr * Etd * cos_phi / (1 - Etau * cos_phi)) /
                ((Etau * cos_phi)**(N - 1) * Etr * Etd * cos_phi + 1)**2 + tau * Etau / (T1**2 * T1_sc * (1 - Etau * cos_phi)) -
                tau * (1 - Etau) * Etau * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)**2))
        tmp2 = ((1 - 2 * Etd + Etr * Etd - (1 - Etau) * (-(Etau * cos_phi)**(N - 1) + 1) * Etr * Etd * cos_phi / (1 - Etau * cos_phi)
                 ) / ((Etau * cos_phi) ** (N - 1) * Etr * Etd * cos_phi + 1) - (1 - Etau) / (1 - Etau * cos_phi)) / (T1**2 * T1_sc)
        tmp3 = - tau * Etau / (T1**2 * T1_sc * (1 - Etau * cos_phi)) + tau * (
            1 - Etau) * Etau * cos_phi / (T1**2 * T1_sc * (1 - Etau * cos_phi)**2)

        for i in range(self.NLL):
            for j in range(self.Nproj):
                n = i * self.Nproj + j + 1

                grad[0, i, j, ...] = M0_sc * sin_phi * \
                    ((cosEtau)**(n - 1) * (Q_F) + F)

                grad[1, i, j, ...] = M0 * M0_sc * ((Etau * cos_phi)**(n - 1) * tmp1 + tau * (
                    Etau * cos_phi)**(n - 1) * (n - 1) * tmp2 + tmp3) * sin_phi

        return np.mean(grad, axis=2)

# return
# np.average(np.array(grad),axis=2,weights=np.tile(self.calc_weights(x[1,:,:,:]),(2,1,1,1,1,1)))


#  cpdef calc_weights(self,DTYPE_t[:,:,::1] x):
#      cdef int i=0,j=0
#      cdef np.ndarray[ndim=3,dtype=DTYPE_t] T1 = np.array(x)*self.T1_sc
#      cdef np.ndarray[ndim=3,dtype=DTYPE_t] w = np.zeros_like(T1)
#      cdef np.ndarray[ndim=3,dtype=DTYPE_t] V = np.ones_like(T1)
#      cdef np.ndarray[ndim=5,dtype=DTYPE_t] result = np.zeros((self.NLL,self.Nproj,self.NSlice,self.dimY,self.dimX),dtype=DTYPE)
#      for i in range(self.NLL):
##           w = 1-np.exp(-(self.tau)/T1)
##           V[~(w==1)] = (1-w[~(w==1)]**self.Nproj)/(1-w[~(w==1)])
#           for j in range(self.Nproj):
#               result[i,j,:,:,:] = np.exp(-(self.td+self.tau*j+self.tau*self.Nproj*i)/T1)/np.exp(-(self.td+self.tau*self.Nproj*i)/T1)
#           result[i,:,:,:,:] = result[i,:,:,:,:]/np.sum(result[i,:,:,:,:],0)
#
#      return np.squeeze(result)


    def plot_unknowns(self, x, dim_2D=False):

        if dim_2D:
            plt.figure(1)
            plt.imshow(np.transpose(np.abs(x[0, ...] * self.M0_sc)))
            plt.pause(0.05)
            plt.figure(2)
            plt.imshow(np.transpose(np.abs(x[1, ...] * self.T1_sc)))
          #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
            plt.pause(0.05)
        else:
            plt.figure(1)
            plt.imshow(np.transpose(
                np.abs(x[0, int(self.NSlice / 2), ...] * self.M0_sc)))
            plt.pause(0.05)
            plt.figure(2)
            plt.imshow(np.transpose(
                np.abs(x[1, int(self.NSlice / 2), ...] * self.T1_sc)))
          #      plt.imshow(np.transpose(np.abs(x[1,0,:,:]*self.model.T1_sc)),vmin=0,vmax=3000)
            plt.pause(0.05)
