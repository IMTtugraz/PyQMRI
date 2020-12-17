import argparse

from pkg_resources import resource_filename
from pyqmri.pyqmri import _setupOCL
from pyqmri._helper_fun import CLProgram as Program

from pyqmri._my_helper_fun.import_data import import_data
from pyqmri._my_helper_fun.display_data import img_montage
from pyqmri._my_helper_fun.export_data import *
from pyqmri._my_helper_fun.recon import *
from pyqmri._my_helper_fun.helpers import *

from pyqmri.softsense import SoftSenseOptimizer
import pyqmri.solver as pyqmrisl


DTYPE = np.complex64
DTYPE_real = np.float32


def _set_output_dir(par, myargs):
    if myargs.outdir == '':
        outdir = Path.cwd()
    else:
        outdir = Path(myargs.outdir)
    outdir = outdir / "SoftSense_out" / myargs.reg_type / time.strftime("%Y-%m-%d_%H-%M-%S")
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    par["outdir"] = outdir


def _setup_par(par, myargs, ksp_data, cmaps):
    ksp_shape = np.shape(ksp_data)
    cmaps_shape = np.shape(cmaps)

    par["C"] = np.require(cmaps, requirements='C')

    par["NScan"] = ksp_shape[0]
    par["dimX"] = ksp_shape[-1]
    par["dimY"] = ksp_shape[-2]
    par["NSlice"] = cmaps_shape[2]
    par["NC"] = cmaps_shape[1]
    par["NMaps"] = cmaps_shape[0]

    par["N"] = par["dimX"]
    par["Nproj"] = par["dimY"]

    # not relevant for my case but necessary for Operator class
    par["unknowns_TGV"] = 0
    par["unknowns_H1"] = 0
    par["unknowns"] = par["NMaps"]
    par["weights"] = np.ones(par["unknowns"], dtype=DTYPE_real)
    par["dz"] = 1

    par["fft_dim"] = (0, 1, 2) if myargs.type == '3D' else (1, 2)
    par["mask"] = np.require(np.ones((par["dimY"], par["dimX"]), dtype=DTYPE_real), requirements='C')
    par["R"] = myargs.acceleration_factor

    par["overlap"] = 0
    par["par_slices"] = par["NSlice"]

    if myargs.streamed:
        par["par_slices"] = int(par["NSlice"] / len(par["num_dev"]))
        par["overlap"] = 1

    _set_output_dir(par, myargs)


def _setup_ss_par(ss_par, myargs):
    ss_par["display_iterations"] = True
    ss_par["accelerated"] = myargs.accelerated
    ss_par["tol"] = 1e-8
    ss_par["stag"] = 1e10
    ss_par["sigma"] = np.float32(1 / np.sqrt(12))
    ss_par["lambd"] = myargs.lamda
    ss_par["alpha0"] = np.sqrt(2)  # 2D --> np.sqrt(2), 3D --> np.sqrt(3)
    ss_par["alpha1"] = 1   # 1
    ss_par["delta"] = 10
    ss_par["gamma"] = 1e-5


def _gen_data_from_imgs(imgs, cmaps, type='2D'):
    result = np.zeros(np.shape(cmaps)).astype(DTYPE)
    if type == '3D':
        for c in range(np.shape(cmaps)[1]):
            for m in range(np.shape(cmaps)[0]):
                if np.ndim(imgs) == 4:
                    result[m, c, ...] = np.fft.ifftshift(np.fft.fftn(
                        np.fft.fftshift(imgs[m, ...] * cmaps[m, c, ...]), norm='ortho'))
                elif np.ndim(imgs) == 3:
                    result[m, c, ...] = np.fft.ifftshift(np.fft.fftn(
                        np.fft.fftshift(imgs * cmaps[m, c, ...]), norm='ortho'))
    elif type == '2D':
        for z in range(np.shape(imgs)[-3]):
            for c in range(np.shape(cmaps)[1]):
                for m in range(np.shape(cmaps)[0]):
                    if np.ndim(imgs) == 4:
                        result[m, c, z, ...] = np.fft.ifftshift(np.fft.fft2(
                            np.fft.fftshift(imgs[m, z, ...] * cmaps[m, c, z, ...]), norm='ortho'))
                    elif np.ndim(imgs) == 3:
                        result[m, c, z, ...] = np.fft.ifftshift(np.fft.fft2(
                            np.fft.fftshift(imgs[z, ...] * cmaps[m, c, z, ...]), norm='ortho'))
    else:
        raise ValueError("Invalid recon type. Must be 2D or 3D.")
    return np.sum(result, axis=0, keepdims=True)


def _pda_soft_sense_solver(myargs, par, ksp, cmaps, imgs, imgs_us):

    ksp = np.require(ksp.astype(DTYPE), requirements='C')
    cmaps = np.require(cmaps.astype(DTYPE), requirements='C')

    if myargs.streamed:
        file = resource_filename(
            'pyqmri', 'kernels/OpenCL_Kernels_streamed.c')
        inp = np.zeros((par["NSlice"], par["NMaps"], par["dimY"], par["dimX"])) + \
              1j * np.zeros((par["NSlice"], par["NMaps"], par["dimY"], par["dimX"]))
    else:
        file = resource_filename(
            'pyqmri', 'kernels/OpenCL_Kernels.c')
        inp = np.zeros((par["NMaps"], par["NSlice"], par["dimY"], par["dimX"])) + \
              1j * np.zeros((par["NMaps"], par["NSlice"], par["dimY"], par["dimX"]))

    inp = inp.astype(DTYPE)

    prg = []
    for j in range(len(par["num_dev"])):
        with open(file) as myfile:
            prg.append(Program(
                par["ctx"][j],
                myfile.read()))

    ss_par = {}
    _setup_ss_par(ss_par, myargs)

    op, _ = pyqmirop.Operator.SoftSenseOperatorFactory(par, prg, DTYPE, DTYPE_real, myargs.streamed)
    grad_op, _ = pyqmirop.Operator.GradientOperatorFactory(par, prg, DTYPE, DTYPE_real, myargs.streamed)
    symgrad_op = pyqmirop.Operator.SymGradientOperatorFactory(par, prg, DTYPE, DTYPE_real, myargs.streamed)

    fval = calculate_cost(myargs, ss_par, imgs_us, np.zeros(inp.shape+(3,)), ksp, cmaps)

    if not myargs.streamed:
        cmaps = cla.to_device(op.queue, cmaps)

    if myargs.linesearch:
        pd = pyqmrisl.PDALSoftSenseBaseSolver.factory(prg, par["queue"], par, ss_par,
                                                      fval, cmaps, (op, grad_op, symgrad_op),
                                                      myargs.reg_type, myargs.streamed)
    else:
        pd = pyqmrisl.PDSoftSenseBaseSolver.factory(prg, par["queue"], par, ss_par,
                                                    fval, cmaps, (op, grad_op, symgrad_op),
                                                    myargs.reg_type, myargs.streamed)

    primal_vars = pd.run(inp=inp.copy(), data=ksp.copy(), iters=1000)["x"]

    if myargs.streamed:
        return np.transpose(primal_vars, (1, 0, 2, 3))

    return primal_vars.get()


def _recon(imgs, ksp_data, cmaps, par, myargs):
    if myargs.streamed:
        data_trans_axes = (2, 0, 1, 3, 4)
        ksp_data = np.transpose(ksp_data, data_trans_axes)
        cmaps = np.transpose(cmaps, data_trans_axes)

    out_undersampled = soft_sense_recon_cl(myargs, par, ksp_data, cmaps)
    out_undersampled_ = out_undersampled.copy()

    if myargs.streamed:
        out_undersampled_ = np.swapaxes(out_undersampled, 0, 1)

    img_montage(sqrt_sum_of_squares(out_undersampled_),
                myargs.type + ' reconstruction of undersampled data with adjoint operator')

    out_pd = _pda_soft_sense_solver(myargs, par, ksp_data, cmaps, imgs, out_undersampled)

    return out_pd


def _start_recon(data, par, myargs):
    optimizer = SoftSenseOptimizer(par,
                                   myargs,
                                   myargs.reg_type,
                                   streamed=myargs.streamed)
    result = optimizer.execute(data.copy())

    return result


def _main(myargs):
    # import kspace and coil sensitivity data
    ksp_data = import_data(myargs.kspfile, 'k-space')[0].astype(DTYPE)
    print('k_space input shape (x, y, z, ncoils): ' + str(np.shape(ksp_data)))

    cs_data = import_data(myargs.csfile, 'cmap')[0]
    cmaps = cs_data.view(DTYPE)
    print('c_maps input shape (nmaps, ncoils, z, y, x) ' + str(cs_data.shape))

    # reorder kspace to (NScan, NC, NSlice, Ny, Nx)
    ksp_data = np.moveaxis(ksp_data, [0, 1, 2, 3], [3, 2, 1, 0])
    ksp_data = np.expand_dims(ksp_data, axis=0)
    ksp_data = prepare_data(ksp_data, rescale=True, recon_type='3D')

    x = phase_recon(ksp_data, cmaps)

    n_slices = x.shape[1]
    if myargs.reco_slices != -1 or myargs.reco_slices > n_slices:
        n_par_slices = myargs.reco_slices
        slice_idx = (int(n_slices/2-n_par_slices/2), int(n_slices/2+n_par_slices/2))
        ksp_data = ksp_data[:, :, slice_idx[0]:slice_idx[-1], :, :]
        cmaps = cmaps[:, :, slice_idx[0]:slice_idx[-1], ...]
        x = x[:, slice_idx[0]:slice_idx[-1], ...]

    img_montage(sqrt_sum_of_squares(x), 'Fully sampled images')

    # setup PyQMRI parameters and PyOCL
    par = {}
    _setupOCL(myargs, par)
    _setup_par(par, myargs, ksp_data, cmaps)

    ksp_data = _gen_data_from_imgs(x, cmaps, myargs.type)
    ksp_data = np.require(prepare_data(ksp_data, recon_type=myargs.type).astype(DTYPE), requirements='C')
    ksp_data = undersample_kspace(par, ksp_data, myargs)

    # x_ssense = _recon(x, ksp_data, cmaps, par, myargs)
    x_ssense = _start_recon(ksp_data, par, myargs)

    x_ssense_ = normalize_imgs(sqrt_sum_of_squares(x_ssense))
    x_ = normalize_imgs(sqrt_sum_of_squares(x))
    x_diff = x_ssense_ - x_

    img_montage(sqrt_sum_of_squares(x_ssense), 'Softsense recon ' + myargs.reg_type)
    img_montage(x_diff, 'Diffs')

    mse, psnr, ssim = calc_image_metrics(x_ssense_, x_)

    # ss_par = {}
    # _setup_ss_par(ss_par, myargs)
    # save_data(x_ssense_, par, ss_par, myargs)
    # save_imgs(x_ssense_, myargs)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    args = argparse.ArgumentParser(
        description="Soft Sense reconstruction.")
    args.add_argument(
      '--recon_type', default='2D', dest='type',
      help='Choose reconstruction type, 2D or 3D')
    args.add_argument(
      '--reg_type', default='NoReg', dest='reg_type',
      help="Choose regularization type (default: without regularization) "
           "options are: 'TGV', 'TV', 'NoReg'")
    args.add_argument(
        '--lambda', default=1, dest='lamda',
        help="Regularization parameter (default: 1)", type=float)
    args.add_argument(
        '--linesearch', default='0', dest='linesearch',
        help="Use PD algorithm with linesearch (default: 0)", type=_str2bool)
    args.add_argument(
        '--accelerated', default='0', dest='accelerated',
        help="Use PD algorithm with adaptive step size (default: 0)", type=_str2bool)
    args.add_argument(
        '--streamed', default='0', dest='streamed', type=_str2bool,
        help='Enable streaming of large data arrays (e.g. >10 slices).')
    args.add_argument(
        '--reco_slices', default='-1', dest='reco_slices', type=int,
        help='Number of slices taken around center for reconstruction (Default to -1, i.e. all slices)')
    args.add_argument(
        '--outdir', default='', dest='outdir', type=str,
        help='Output directory.')

    args = args.parse_args()

    args.trafo = False
    args.use_GPU = True
    # args.streamed = False
    args.devices = -1
    args.kspfile = Path.cwd() / 'data_soft_sense_test' / 'kspace.mat'
    args.csfile = Path.cwd() / 'data_soft_sense_test' / 'sensitivities_ecalib.mat'

    #args.type = '2D'
    #args.reg_type = 'TGV'  # 'NoReg', 'TV', or 'TGV'
    #args.reco_slices = 4
    args.linesearch = False
    args.accelerated = False
    #args.lamda = 1.0
    args.dim_us = 'y'
    args.acceleration_factor = 4

    _main(args)
