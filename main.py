import argparse
import sys

import pyopencl.array as cla
import pyqmri.operator as pyqmirop
import pyqmri.solver as pyqmrisl

from pyqmri._my_helper_fun.import_data import import_data
from pyqmri._my_helper_fun.display_data import *
from pyqmri._my_helper_fun.export_data import *
from pyqmri._my_helper_fun.recon import phase_recon_cl_3d, soft_sense_recon_cl, calculate_cost
from pyqmri._my_helper_fun.helpers import *
from pyqmri.pyqmri import _setupOCL
from pkg_resources import resource_filename
from pyqmri._helper_fun import CLProgram as Program

DTYPE = np.complex64
DTYPE_real = np.float32


def _setup_par(par, myargs, ksp_data):
    ksp_shape = np.shape(ksp_data)
    par["dimX"] = ksp_shape[-1]
    par["dimY"] = ksp_shape[-2]
    par["NSlice"] = ksp_shape[-3]
    par["NC"] = ksp_shape[1]
    par["NScan"] = ksp_shape[0]
    par["NMaps"] = 2

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

    par["overlap"] = 0
    par["par_slices"] = par["NSlice"]

    if myargs.streamed:
        par["par_slices"] = int(par["NSlice"] / (2 * len(par["num_dev"])))
        par["overlap"] = 1


def _setup_irgn_par(irgn_par, myargs):
    irgn_par["display_iterations"] = False
    irgn_par["accelerated"] = myargs.accelerated
    irgn_par["tol"] = 1e-8
    irgn_par["stag"] = 1e10
    irgn_par["sigma"] = np.float32(1 / np.sqrt(12))
    irgn_par["lambd"] = myargs.lamda
    irgn_par["alpha0"] = np.sqrt(2)  # 2D --> np.sqrt(2), 3D --> np.sqrt(3)
    irgn_par["alpha1"] = 1   # 1
    irgn_par["delta"] = 10
    irgn_par["gamma"] = 1e-5


def _power_iterations(par, x, cmaps, op, iters=100):
    queue = par["queue"][0]
    x = x.astype(DTYPE)
    x = cla.to_device(queue, x)

    if len(cmaps) > 0:
        cmaps = cmaps.astype(DTYPE)
        cmaps = cla.to_device(queue, cmaps)
        y = op.adjoop([op.fwdoop([x, cmaps]), cmaps]).get()
    else:
        y = op.adjoop(op.fwdoop(x)).get()

    for i in range(iters):
        y_norm = np.linalg.norm(y)
        x = y / y_norm if y_norm != 0 else y
        x = cla.to_device(queue, x)
        y = op.adjoop([op.fwdoop([x, cmaps]), cmaps]).get() if len(cmaps) > 0 else op.adjoop(op.fwdoop(x)).get()
        x = x.get()
        l1 = np.vdot(y, x)

    return np.sqrt(np.max(np.abs(l1)))


def _calc_step_size(args, par, ksp, cmaps):

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

    op = pyqmirop.OperatorSoftSense(par, prg)
    grad_op = pyqmirop.OperatorFiniteGradient(par, prg)
    symgrad_op = pyqmirop.OperatorFiniteSymGradient(par, prg)

    x = np.random.randn(1, par["NSlice"], par["dimY"], par["dimX"]) + \
        1j * np.random.randn(1, par["NSlice"], par["dimY"], par["dimX"])

    opnorm_1 = _power_iterations(par, x, cmaps[0], op)

    x = np.random.randn(1, par["NSlice"], par["dimY"], par["dimX"]) + \
        1j * np.random.randn(1, par["NSlice"], par["dimY"], par["dimX"])

    opnorm_2 = _power_iterations(par, x, cmaps[1], op)

    opnorm_grad = _power_iterations(par, x, [], grad_op)

    x_symgrad = np.random.randn(1, par["NSlice"], par["dimY"], par["dimX"], 4) + \
        1j * np.random.randn(1, par["NSlice"], par["dimY"], par["dimX"], 4)

    opnorm_symgrad = _power_iterations(par, x_symgrad, [], symgrad_op)

    K_ssense = np.array([opnorm_1, opnorm_2])
    K_ssense_tv = np.array([[opnorm_1,      opnorm_2],
                            [opnorm_grad,   0],
                            [0,             opnorm_grad]])

    K_ssense_tgv = np.array([[opnorm_1,     opnorm_2,       0,              0],
                             [opnorm_grad,  0,              1,              0],
                             [0,            0,              opnorm_symgrad, 0],
                             [0,            opnorm_grad,    0,              1],
                             [0,            0,              0,              opnorm_symgrad]])

    if args.reg_type == '':
        tau = 1 / np.sqrt(np.vdot(K_ssense, K_ssense))
    if args.reg_type == 'TV':
        tau = 1 / np.sqrt(np.vdot(K_ssense_tv, K_ssense_tv))
    if args.reg_type == 'TGV':
        tau = 1 / np.sqrt(np.vdot(K_ssense_tgv, K_ssense_tgv))
    sigma = tau
    return tau, sigma


def _pda_soft_sense_solver(myargs, par, ksp, cmaps, imgs, imgs_us):

    #tau, sigma =  _calc_step_size(myargs, par, ksp, cmaps)

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

    irgn_par = {}
    _setup_irgn_par(irgn_par, myargs)

    op = pyqmirop.Operator.SoftSenseOperatorFactory(par, prg, DTYPE, DTYPE_real, myargs.streamed)
    grad_op = pyqmirop.Operator.GradientOperatorFactory(par, prg, DTYPE, DTYPE_real, myargs.streamed)
    symgrad_op = pyqmirop.Operator.SymGradientOperatorFactory(par, prg, DTYPE, DTYPE_real, myargs.streamed)

    fval = calculate_cost(myargs, irgn_par, imgs_us, np.zeros(inp.shape+(3,)), ksp, cmaps)

    if not myargs.streamed:
        cmaps = cla.to_device(op.queue, cmaps)

    if myargs.linesearch:
        pd = pyqmrisl.PDALSoftSenseBaseSolver.factory(prg, par["queue"], par, irgn_par,
                                                    fval, cmaps, (op, grad_op, symgrad_op), None, myargs.reg_type)
    else:
        pd = pyqmrisl.PDSoftSenseBaseSolver.factory(prg, par["queue"], par, irgn_par,
                                                    fval, cmaps, (op, grad_op, symgrad_op),
                                                    None, myargs.reg_type, SMS=False, streamed=myargs.streamed)

    primal_vars = pd.run(inp=inp.copy(), data=ksp.copy(), iters=1000)["x"]

    if myargs.streamed:
        return np.transpose(primal_vars, (1, 0, 2, 3))

    return primal_vars.get()


def _3d_recon(imgs, ksp_data, cmaps, par, myargs):
    # if myargs.undersampling:
    #     ksp_data = undersample_kspace(par, ksp_data, myargs)

    if myargs.streamed:
        data_trans_axes = (2, 0, 1, 3, 4)
        ksp_data = np.transpose(ksp_data, data_trans_axes)
        cmaps = np.transpose(cmaps, data_trans_axes)

    out_undersampled = soft_sense_recon_cl(myargs, par, ksp_data, cmaps)

    if myargs.streamed:
        out_undersampled_ = np.moveaxis(out_undersampled, [0, 1], [1, 0])
        img_montage(sum_of_squares(out_undersampled_), '3D reconstruction of undersampled data with adjoint operator')
    else:
        img_montage(sum_of_squares(out_undersampled), '3D reconstruction of undersampled data with adjoint operator')

    out_pd = _pda_soft_sense_solver(myargs, par, ksp_data, cmaps, imgs, out_undersampled)

    x_orig = normalize_imgs(sum_of_squares(imgs))
    x_ssense = normalize_imgs(sum_of_squares(out_pd))
    x_diff = normalize_imgs(x_orig - x_ssense)

    return x_ssense, x_orig, x_diff


def _2d_recon(imgs, ksp_data, cmaps, par, myargs):

    ksp_data2d = gen_2ddata_from_imgs(imgs, cmaps)

    if myargs.undersampling:
        ksp_data2d = undersample_kspace(par, ksp_data2d, myargs)

    # Select only several slices (performance/duration)
    # n_slice = 4
    # par["NSlice"] = n_slice
    # par["par_slices"] = int(par["NSlice"] / (2 * len(par["num_dev"]))) if myargs.streamed else n_slice
    #
    # ksp_data2d = ksp_data2d[:, :, 21:21+n_slice, :, :]
    # cmaps = cmaps[:, :, 21:21+n_slice, ...]
    # imgs = imgs[:, 21:21+n_slice, ...]

    # ksp_data2d = np.repeat(ksp_data2d[:, :, [0], :, :], n_slice, axis=2)
    # cmaps = np.repeat(cmaps[:, :, [0], :, :], n_slice, axis=2)
    # imgs = np.repeat(imgs[:, [0], :, :], n_slice, axis=1)

    if myargs.streamed:
        data_trans_axes = (2, 0, 1, 3, 4)
        ksp_data2d = np.transpose(ksp_data2d, data_trans_axes)
        cmaps = np.transpose(cmaps, data_trans_axes)

    out_undersampled = soft_sense_recon_cl(myargs, par, ksp_data2d, cmaps)

    if myargs.streamed:
        out_undersampled_ = np.moveaxis(out_undersampled, [0, 1], [1, 0])
        img_montage(sum_of_squares(out_undersampled_), '2D reconstruction of undersampled data with adjoint operator')
    else:
        img_montage(sum_of_squares(out_undersampled), '2D reconstruction of undersampled data with adjoint operator')

    out_pd = _pda_soft_sense_solver(myargs, par, ksp_data2d, cmaps, imgs, out_undersampled)

    x_orig = normalize_imgs(sum_of_squares(imgs))
    x_ssense = normalize_imgs(sum_of_squares(out_pd))
    x_diff = x_orig - x_ssense

    return x_ssense, x_orig, x_diff


def _main(myargs):
    # import kspace and coil sensitivity data
    # make sure to convert to single precision
    ksp_data = import_data(myargs.kspfile, 'k-space')[0].astype(DTYPE)
    print('k_space size (x, y, z, ncoils): ' + str(np.shape(ksp_data)))

    cs_data = import_data(myargs.csfile, 'cmap')[0]
    print('c_maps size (nmaps, ncoils, z, y, x) ' + str(cs_data.shape))

    # cIFFT supports transform sizes that are powers of 2, 3, 5, 7. Thus the vector length has to be a combination
    # of powers of these numbers e.g. 3**2 * 5**4
    # NC * NSlice = 52 * 51 = 2652 --> not valid --> reduce NSlice to 50
    # also reorder kspace to (NScan (NMaps), NC, NSlice, y, x)
    ksp_data = np.moveaxis(ksp_data, [0, 1, 2, 3], [3, 2, 1, 0])
    ksp_data = ksp_data[:, :50, :, :]  # without scaling --> numerical errors --> no reasonable solution
    # rescale with sqrt(X*Y*Z)
    ksp_data = prepare_data(ksp_data, rescale=True, recon_type='3D').astype(DTYPE)
    ksp_data = np.expand_dims(ksp_data, axis=0)
    cs_data = cs_data[:, :, :50, :, :]
    cmaps = cs_data.view(DTYPE)

    # setup PyQMRI parameters and PyOCL
    par = {}
    _setupOCL(myargs, par)
    _setup_par(par, myargs, ksp_data)

    imgs = phase_recon_cl_3d(ksp_data, cmaps, par)
    img_montage(sum_of_squares(imgs), 'Phase sensitive reconstruction 3D')

    if myargs.type == '2D':
        out_ssense, out_orig, out_diff = _2d_recon(imgs, ksp_data, cmaps, par, myargs)
    elif myargs.type == '3D':
        out_ssense, out_orig, out_diff = _3d_recon(imgs, ksp_data, cmaps, par, myargs)
    else:
        raise ValueError("Invalid recon type. Must be 2D or 3D.")

    img_montage(out_orig, 'Original selected images')
    img_montage(out_ssense, 'Softsense recon ' + myargs.reg_type)
    img_montage(out_diff, 'Diffs')

    mse, psnr, ssim = calc_image_metrics(out_ssense, out_orig)

    irgn_par = {}
    _setup_irgn_par(irgn_par, myargs)
    save_data(out_ssense, par, irgn_par, myargs)
    # save_data(out_orig, par, irgn_par, myargs, filename='fully_sampled_recon', ds_name='dataset', store_attrs=False)
    save_imgs(out_ssense, myargs)
    save_imgs(out_diff, myargs, 'diffs')


if __name__ == '__main__':

    args = argparse.ArgumentParser(
        description="Soft Sense reconstruction.")
    args.add_argument(
      '--recon_type', default='2D', dest='type',
      help='Choose reconstruction type, 2D or 3D')
    args.add_argument(
      '--reg_type', default='', dest='reg_type',
      help="Choose regularization type (default: without regularization) "
           "options are: 'TGV', 'TV', ''")
    args.add_argument(
        '--lambda', default=1, dest='lamda',
        help="Regularization parameter (default: 1)", type=float
    )

    args = args.parse_args()

    args.trafo = False
    args.use_GPU = True
    args.streamed = True
    args.devices = -1
    args.kspfile = Path.cwd() / 'data_soft_sense_test' / 'kspace.mat'
    args.csfile = Path.cwd() / 'data_soft_sense_test' / 'sensitivities_ecalib.mat'

    args.type = '2D'
    # args.reg_type = 'TV'  # '', 'TV', or 'TGV'
    args.linesearch = False
    args.accelerated = False
    # args.lamda = 1.0
    args.undersampling = True
    args.dim_us = 'y'
    args.acceleration_factor = 4

    _main(args)
