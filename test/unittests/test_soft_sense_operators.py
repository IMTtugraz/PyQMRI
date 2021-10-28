try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
import h5py
import pyopencl.array as cla
import pyqmri.operator as pyqmirop

from pyqmri.pyqmri import _setupOCL
from pkg_resources import resource_filename
from pyqmri._helper_fun import CLProgram as Program
from pathlib import Path


DTYPE = np.complex64
DTYPE_real = np.float32


class Args:
    pass


def _setup_par(par):
    par["dimX"] = 160
    par["dimY"] = 160
    par["NSlice"] = 180
    par["NC"] = 32
    par["NScan"] = 1
    par["NMaps"] = 2

    par["N"] = par["dimX"]
    par["Nproj"] = par["dimY"]
    par["unknowns_TGV"] = 0
    par["unknowns_H1"] = 0
    par["unknowns"] = 2
    par["dz"] = 1

    par["overlap"] = 0
    par["par_slices"] = 18 # par["NSlice"]

    par["DTYPE_real"] = DTYPE_real
    par["DTYPE"] = DTYPE

    par["fft_dim"] = (-2, -1)
    par["mask"] = np.ones((1, 32, 180, 160, 160), dtype=DTYPE_real)
    # par["mask"][..., ::2, :] = 0


class OperatorSoftSenseTest(unittest.TestCase):
    def setUp(self):
        args = Args
        args.trafo = False
        args.use_GPU = True
        args.streamed = False
        args.devices = 0

        par = {}
        _setupOCL(args, par)
        _setup_par(par)

        if DTYPE == np.complex128:
            file = open(
                    resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_double.c'))
        else:
            file = open(
                    resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels.c'))
        prg = Program(
            par["ctx"][0],
            file.read())
        file.close()

        maskfile = Path.cwd() / 'data' / 'R4' / 'mask_a.mat'

        with h5py.File(maskfile, 'r') as f:
            mask = np.array(f.get('mask_a'))
        f.close()

        par['mask'] = np.require(mask.astype(par["DTYPE_real"]), requirements='C')
        par['mask'] = np.require(np.fft.fftshift(par['mask'], axes=par["fft_dim"]), requirements='C')

        self.op = pyqmirop.OperatorSoftSense(par, prg)

        self.opinfwd = np.random.randn(par["NMaps"], par["NSlice"], par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NMaps"], par["NSlice"],  par["dimY"], par["dimX"])

        self.opinadj = np.random.randn(par["NScan"], par["NC"], par["NSlice"], par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NScan"], par["NC"], par["NSlice"], par["dimY"], par["dimX"])

        self.C = np.random.randn(par["NMaps"], par["NC"], par["NSlice"], par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NMaps"], par["NC"], par["NSlice"], par["dimY"], par["dimX"])

        self.C = self.C.astype(DTYPE)
        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)
        self.queue = par["queue"][0]
        self.coil_buf = cla.to_device(self.queue, self.C)

    def test_outofplace(self):
        inpfwd = cla.to_device(self.queue, self.opinfwd)
        inpadj = cla.to_device(self.queue, self.opinadj)

        outfwd = self.op.fwdoop([inpfwd, self.coil_buf])
        outadj = self.op.adjoop([inpadj, self.coil_buf])

        outfwd = outfwd.get()
        outadj = outadj.get()

        a = np.vdot(outfwd.flatten(), self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(), outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=6)

    def test_inplace(self):
        inpfwd = cla.to_device(self.queue, self.opinfwd)
        inpadj = cla.to_device(self.queue, self.opinadj)

        outfwd = cla.zeros_like(inpadj)
        outadj = cla.zeros_like(inpfwd)

        self.op.fwd(outfwd, [inpfwd, self.coil_buf])
        self.op.adj(outadj, [inpadj, self.coil_buf])

        outfwd = outfwd.get()
        outadj = outadj.get()

        a = np.vdot(outfwd.flatten(), self.opinadj.flatten()) / self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(), outadj.flatten()) / self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=6)


class OperatorSoftSenseStreamedTest(unittest.TestCase):
    def setUp(self):
        args = Args
        args.use_GPU = True
        args.streamed = True
        args.devices = -1

        par = {}
        _setupOCL(args, par)
        _setup_par(par)

        if DTYPE == np.complex128:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_double_streamed.c')
        else:
            file = resource_filename(
                        'pyqmri', 'kernels/OpenCL_Kernels_streamed.c')

        prg = []
        for j in range(len(par["ctx"])):
          with open(file) as myfile:
            prg.append(Program(
                par["ctx"][j],
                myfile.read()))

        par["par_slices"] = 1

        maskfile = Path.cwd() / 'data' / 'R4' / 'mask_a.mat'

        with h5py.File(maskfile, 'r') as f:
            mask = np.array(f.get('mask_a'))
        f.close()

        par['mask'] = np.require(mask.astype(par["DTYPE_real"]), requirements='C')
        par['mask'] = np.require(np.fft.fftshift(par['mask'], axes=par["fft_dim"]), requirements='C')

        self._data_trans_axes = (2, 0, 1, 3, 4)
        par["mask"] = np.transpose(np.expand_dims(par["mask"], axis=0), self._data_trans_axes)
        self.mask = np.require(par["mask"], requirements='C', dtype=DTYPE_real)

        self.op = pyqmirop.OperatorSoftSenseStreamed(
            par, prg,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real,
            trafo=False
        )

        self.opinfwd = np.random.randn(par["NSlice"], par["NMaps"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NSlice"], par["NMaps"],
                                 par["dimY"], par["dimX"])
        self.opinadj = np.random.randn(par["NSlice"], par["NScan"], par["NC"],
                                       par["dimY"], par["dimX"]) +\
            1j * np.random.randn(par["NSlice"], par["NScan"], par["NC"],
                                 par["dimY"], par["dimX"])
        self.C = np.random.randn(par["NSlice"], par["NMaps"], par["NC"],
                                 par["dimY"], par["dimX"]) + \
            1j * np.random.randn(par["NSlice"], par["NMaps"], par["NC"],
                                 par["dimY"], par["dimX"])

        self.opinfwd = self.opinfwd.astype(DTYPE)
        self.opinadj = self.opinadj.astype(DTYPE)
        self.C = self.C.astype(DTYPE)

    def test_adj_outofplace(self):

        outfwd = self.op.fwdoop([[self.opinfwd, self.C, self.mask]])
        outadj = self.op.adjoop([[self.opinadj, self.C, self.mask]])

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=6)

    def test_adj_inplace(self):

        outfwd = np.zeros_like(self.opinadj)
        outadj = np.zeros_like(self.opinfwd)

        self.op.fwd([outfwd], [[self.opinfwd, self.C, self.mask]])
        self.op.adj([outadj], [[self.opinadj, self.C, self.mask]])

        a = np.vdot(outfwd.flatten(),
                    self.opinadj.flatten())/self.opinadj.size
        b = np.vdot(self.opinfwd.flatten(),
                    outadj.flatten())/self.opinadj.size

        print("Adjointness: %.2e +1j %.2e" % ((a - b).real, (a - b).imag))

        self.assertAlmostEqual(a, b, places=6)


if __name__ == '__main__':
    unittest.main()


