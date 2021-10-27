import argparse
import h5py
import math
import numpy as np
import os
import sys
import time

from scipy.io import loadmat
from tkinter import filedialog
from tkinter import Tk

from pyqmri.pyqmri import _str2bool
from pyqmri.pyqmri import _setupOCL
from pyqmri._helper_fun import _utils as utils
from pyqmri.pdsose import SoftSenseOptimizer


# def _check_data_shape(myargs, data):
#     gpyfft_primes = [2, 3, 5, 7, 11, 13]
#     z, y, x = np.shape(data)[-3:]
#     reco_slices = myargs.reco_slices
#     while reco_slices > 0:
#         data_prime_factors = utils.prime_factors(reco_slices*y*x)
#         l = [i for i in data_prime_factors if i not in gpyfft_primes]
#         if not l:
#             break
#         else:
#             reco_slices -= 1
#
#     if myargs.reco_slices != reco_slices:
#         print("Required to reduce number of slices. Reducing slices to %i" %reco_slices)
#
#     myargs.reco_slices = reco_slices


def _fft_shift_data(ksp, recon_type='3D'):
    shape = np.shape(ksp)
    fft_shift_dim = (-2, -1)
    nc, z, y, x = shape[-4:]
    check = np.ones_like(ksp)
    check[..., 1::2] = -1
    check[..., ::2, :] *= -1
    if recon_type == '3D':
        check[..., ::2, :, :] *= -1
        fft_shift_dim = (-3, -2, -1)

    result = ksp * check
    for n in range(nc):
        result[:, n, ...] = np.fft.ifftshift(result[:, n, ...], axes=fft_shift_dim)

    return result


def _preprocess_data(myargs, par, kspace, cmaps):
    kspace = _fft_shift_data(kspace, recon_type=myargs.type)

    if myargs.type == '3D':
        kspace = np.fft.ifft(kspace, axis=-1, norm='ortho')
        kspace = np.require(
            np.swapaxes(kspace, -1, -3),
            dtype=par["DTYPE"], requirements='C')
        cmaps = np.require(
            np.swapaxes(cmaps, -1, -3),
            dtype=par["DTYPE"], requirements='C')

    # _check_data_shape(myargs, kspace)  # TODO: check data shape if suitable for clFFT

    if myargs.reco_slices > 0:
        slice_idx = (int(par["NSlice"] / 2) - int(math.floor(par["NSlice"] / 2)),
                     int(par["NSlice"] / 2) + int(math.ceil(par["NSlice"] / 2)))

        kspace = np.require(kspace[..., slice_idx[0]:slice_idx[-1], :, :],
                            requirements='C').astype(par["DTYPE"])
        cmaps = np.require(cmaps[..., slice_idx[0]:slice_idx[-1], :, :],
                           requirements='C').astype(par["DTYPE"])

    return kspace, cmaps


def _setup_par(par, myargs, ksp_data, cmaps, mask):
    ksp_shape = np.shape(ksp_data)
    cmaps_shape = np.shape(cmaps)

    par["C"] = np.require(cmaps, requirements='C').astype(par["DTYPE"])

    par["NScan"] = ksp_shape[0] if len(ksp_shape) == 5 else 1

    par["dimX"] = ksp_shape[-1]
    par["dimY"] = ksp_shape[-2]
    par["NSlice"] = ksp_shape[-3]

    par["NMaps"] = cmaps_shape[0]
    par["NC"] = cmaps_shape[1]

    par["N"] = par["dimX"]
    par["Nproj"] = par["dimY"]

    par["unknowns_TGV"] = par["NMaps"]
    par["unknowns"] = par["NMaps"]
    par["weights"] = np.ones(par["unknowns"], dtype=par["DTYPE_real"])
    par["dz"] = 1

    # not relevant for this case but necessary for Operator class
    par["unknowns_H1"] = 0

    par["fft_dim"] = (-2, -1)

    par["mask"] = np.require(mask, requirements='C', dtype=par["DTYPE_real"])

    par["R"] = 4  # TODO: replace with actual estimation

    par["overlap"] = 0
    par["par_slices"] = par["NSlice"]

    if myargs.streamed:
        if myargs.reco_slices == -1:
            par["par_slices"] = int(par["dimX"] / (2 * len(par["num_dev"])))
        else:
            par["par_slices"] = int(myargs.reco_slices / (2 * len(par["num_dev"])))
        par["overlap"] = 1


def _set_output_dir(myargs, par, name):
    if myargs.outdir == '':
        outdir = os.sep.join(name.split(os.sep)[:-1]) + os.sep + \
            "Soft_SENSE_out" + os.sep + \
            time.strftime("%Y-%m-%d_%H-%M-%S") + os.sep
    else:
        outdir = myargs.outdir + os.sep + "Soft_SENSE_out" + \
            os.sep + os.sep.join(name.split(os.sep)[:-1]) + os.sep + \
            time.strftime("%Y-%m-%d_%H-%M-%S") + os.sep
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    par["outdir"] = outdir


def _read_input(file, myargs, par, set_outdir=False):
    if file == '':
        select_file = True
        while select_file is True:
            root = Tk()
            root.withdraw()
            root.update()
            file_fd = filedialog.askopenfilename()
            root.destroy()
            if file_fd and \
               not file_fd.endswith((('.h5'), ('.hdf5'), ('.mat'))):
                print("Please specify a h5 or mat file. Press cancel to exit.")
            elif not file_fd:
                print("Exiting...")
                sys.exit()
            else:
                select_file = False
                file = file_fd
    else:
        if not file.endswith((('.h5'), ('.hdf5'), ('.mat'))):
            print("Please specify a h5 or mat file. ")
            sys.exit()

    fname = os.path.normpath(file)
    par["fname"] = fname.split(os.sep)[-1]

    if set_outdir:
        _set_output_dir(myargs, par, fname)

    if file.endswith((('.h5'), ('.hdf5'))):
        datafile = h5py.File(file, 'a')
    else:
        try:
            datafile = loadmat(file)
        except NotImplementedError:
            datafile = h5py.File(file, 'a')
        except:
            ValueError("File {} not readable...".format(file))
            sys.exit()

    return datafile


def _cvt_struct_to_complex(data, double_precision=False):
    data_type = np.complex64
    if double_precision:
        data_type = np.complex128
    return (data['real'] + 1j * data['imag']).astype(data_type)


def _read_data_from_file(file, double_precision=False):
    # TODO: check if is .mat or h5 format and handle appropriately
    data = np.array(file.get(list(file.keys())[0]))
    if len(data.dtype) > 1:
        data = _cvt_struct_to_complex(data, double_precision)

    return data


def _start_recon(myargs):
    # Create par struct to store relevant parameters for reconstruction
    par = {}

    ###############################################################################
    # Define precision ############################################################
    ###############################################################################
    if myargs.double_precision:
        par["DTYPE"] = np.complex128
        par["DTYPE_real"] = np.float64
    else:
        par["DTYPE"] = np.complex64
        par["DTYPE_real"] = np.float32

    data = _read_data_from_file(_read_input(myargs.file, myargs, par, set_outdir=True), myargs.double_precision)
    cmaps = _read_data_from_file(_read_input(myargs.cmaps, myargs, par), myargs.double_precision)
    mask = _read_data_from_file(_read_input(myargs.mask, myargs, par), myargs.double_precision)

    data = data * 1e7

    _setupOCL(myargs, par)

    data, cmaps = _preprocess_data(myargs, par, data, cmaps)

    _setup_par(par, myargs, data, cmaps, mask)

    optimizer = SoftSenseOptimizer(par,
                                   myargs,
                                   myargs.reg_type,
                                   streamed=myargs.streamed,
                                   DTYPE=par["DTYPE"],
                                   DTYPE_real=par["DTYPE_real"])
    result = optimizer.execute(data.copy())

    del optimizer



def run(recon_type='3D',
        reg_type='TGV',
        adapt_stepsize=True,
        reco_slices=-1,
        streamed=False,
        par_slices=1,
        devices=-1,
        dz=1,
        weights=-1,
        data='',
        cmaps='',
        mask='',
        config='default',
        outdir='',
        double_precision=False):
    """
    Start a Soft SENSE reconstruction

    Start a Soft SENSE reconstruction
    If no data path is given, a file dialog can be used to select data,
    coil sensitivities and sampling mask (binary undersampling pattern)
    at start up.

    If no config file is passed, a default one will be generated in the
    current folder, the script is run in.

    Parameters
    ----------
      recon_type : str, 3D
        2D or 3D. If 3D is selected the FFT is computed along the fully
        sampled dimension.
      reg_type : str, TGV
        TGV or TV, defaults to TGV
      adapt_stepsize : bool, True
        Primal dual algorithm with adaptive stepsize is used for reconstruction.
        If false, step sizes are estimated using power method.
      reco_slices : int, -1
        The number of slices to reconsturct. Slices are picked symmetrically
        from the volume center. Pass -1 to select all slices available.
        Defaults to -1
      streamed : bool, False
        Toggle between streaming slices to the GPU (1) or computing
        everything with a single memory transfer (0). Defaults to 0
      par_slices : int, 1
        Number of slices per streamed package. Volume devided by GPU's and
        par_slices must be an even number! Defaults to 1
      devices : list of int, 0
        The device ID of device(s) to use for streaming/reconstruction
      dz : float, 1
        Ratio of physical Z to X/Y dimension. X/Y is assumed to be isotropic.
      outdir : str, ''
        Output directory. Defaults to the location of the input file.
      weights : list of float, -1
        Optional weights for each unknown. Defaults to -1, i.e. no additional
        weights are used.
      data : str, ''
        The path to the .h5 file containing the data to reconstruct.
        If left empty, a GUI will open and asks for data file selection. This
        is also the default behaviour.
      cmaps : str, ''
        The path to the .h5 file containing the estimated coil sensitivities.
        If left empty, a GUI will open and asks for data file selection. This
        is also the default behaviour.
      mask : str, ''
        The path to the .h5 file containing the binary mask representing the
        sampling pattern.
        If left empty, a GUI will open and asks for data file selection. This
        is also the default behaviour.
      config : str, default
        The path to the config file used for the Soft SENSE PD reconstruction. If
        not specified the default config file will be used. If no default
        config file is present in the current working directory one will be
        generated.
      double_precision : bool, False
        Enable double precission computation.
    """
    params = [('--recon_type', str(recon_type)),
              ('--reg_type', str(reg_type)),
              ('--adapt_stepsize', str(adapt_stepsize)),
              ('--reco_slices', str(reco_slices)),
              ('--streamed', str(streamed)),
              ('--devices', str(devices)),
              ('--dz', str(dz)),
              ('--weights', str(weights)),
              ('--par_slices', str(par_slices)),
              ('--data', str(data)),
              ('--cmaps', str(cmaps)),
              ('--mask', str(mask)),
              ('--config', str(config)),
              ('--OCL_GPU', "True"),
              ('--outdir', str(outdir)),
              ('--double_precision', str(double_precision))
              ]

    sysargs = sys.argv[1:]
    for par_name, par_value in params:
        if par_name not in sysargs:
            sysargs.append(par_name)
            sysargs.append(par_value)
    argsrun, unknown = _parse_arguments(sysargs)
    if unknown:
        print("Unknown command line arguments passed: " + str(unknown) + "."
              " These will be ignored for reconstruction.")
    _start_recon(argsrun)


def _parse_arguments(args):
    argpar = argparse.ArgumentParser(
        description="3D Cartesian Soft-SENSE reconstruction"
                    " through Primal-dual algorithm"
                    " with TV or TGV regularization."
    )
    argpar.add_argument(
      '--recon_type', default='3D', dest='type',
      help="Choose reconstruction type, 2D or 3D. "
           "Default is 3D.")
    argpar.add_argument(
      '--reg_type', default='NoReg', dest='reg_type',
      help="Choose regularization type "
           "options are: 'TGV', 'TV'")
    argpar.add_argument(
        '--adapt_stepsize', default='0', dest='adapt_stepsize', type=_str2bool,
        help='Enable accelerated optimization by adaptive step size computation.')
    argpar.add_argument(
        '--streamed', default='0', dest='streamed', type=_str2bool,
        help='Enable streaming of large data arrays (e.g. >10 slices).')
    argpar.add_argument(
        '--reco_slices', default='-1', dest='reco_slices', type=int,
        help='Number of slices taken around center for reconstruction (Default to -1, i.e. all slices)')
    argpar.add_argument(
      '--par_slices', dest='par_slices', type=int,
      help='number of slices per package. Volume devided by GPU\'s and'
           ' par_slices must be an even number!')
    argpar.add_argument(
      '--devices', dest='devices', type=int,
      help="Device ID of device(s) to use for streaming. "
           "-1 selects all available devices", nargs='*')
    argpar.add_argument(
      '--OCL_GPU', dest='use_GPU', type=_str2bool,
      help="Select if CPU or GPU should be used as OpenCL platform. "
           "Defaults to GPU (1). CAVE: CPU FFT not working")
    argpar.add_argument(
      '--dz', dest='dz', type=float,
      help="Ratio of physical Z to X/Y dimension. "
           "X/Y is assumed to be isotropic. Defaults to  1")
    argpar.add_argument(
      '--weights', dest='weights', type=float,
      help="Ratio of unkowns to each other. Defaults to 1. "
           "If passed, needs to be in the same size as the number of unknowns",
           nargs='*')
    argpar.add_argument(
      '--data', dest='file',
      help="Full path to input data. "
           "If not provided, a file dialog will open.")
    argpar.add_argument(
      '--cmaps', dest='cmaps',
      help="Full path to coil sensitivity maps. "
           "If not provided, a file dialog will open.")
    argpar.add_argument(
      '--mask', dest='mask',
      help="Full path to binary mask. "
           "If not provided, a file dialog will open.")
    argpar.add_argument(
        '--config', dest='config',
        help='Name of config file to use (assumed to be in the same folder). \
              If not specified, use default parameters.')
    argpar.add_argument(
      '--double_precision', dest='double_precision', type=_str2bool,
      help="Switch between single (False, default) and double "
           "precision (True). Usually, single precision gives high enough "
           "accuracy.")
    argpar.add_argument(
        '--outdir', default='', dest='outdir', type=str,
        help='The path of the output directory. Default is to the '
             'location of the input file directory')

    return argpar.parse_known_args(args)


if __name__ == '__main__':
    run()
