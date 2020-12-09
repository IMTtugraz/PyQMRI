import time
import pyopencl.array as cla
import pyqmri.operator as pyqmirop

from pyqmri._my_helper_fun.display_data import *
from pyqmri._my_helper_fun.export_data import *
from pyqmri._my_helper_fun.helpers import create_mask
from pyqmri.transforms import PyOpenCLnuFFT
from pkg_resources import resource_filename
from pyqmri._helper_fun import CLProgram as Program

DTYPE = np.complex64
DTYPE_real = np.float32


def phase_recon_cl(x, cmap, par):
    nctmp = par["NC"]
    par["NC"] = 1
    fft = PyOpenCLnuFFT.create(par["ctx"][0], par["queue"][0], par, fft_dim=par["fft_dim"])
    par["NC"] = nctmp

    size = np.shape(x)
    result = np.zeros(
        (par["NScan"], par["NC"], par["NSlice"],
         par["dimY"], par["dimX"]), dtype=DTYPE)
    tmp_result = cla.empty(
        fft.queue,
        (1, 1, par["NSlice"], par["dimY"], par["dimX"]),
        dtype=DTYPE)

    start = time.time()
    for n in range(size[0]):
        for c in range(size[1]):
            clainput = cla.to_device(fft.queue,
                                    np.require(
                                        x[n, c, ...][None, None, ...],
                                        requirements='C'))
            fft.FFTH(tmp_result, clainput).wait()
            result[n, c, ...] = np.fft.fftshift(np.squeeze(tmp_result.get()), axes=par["fft_dim"])
    print("FT took %f s" % (time.time() - start))

    return np.stack((np.require(np.sum(result[0, ...] * np.conj(cmap[0, ...]), axis=0), requirements='C'),
            np.require(np.sum(result[0, ...] * np.conj(cmap[1, ...]), axis=0), requirements='C')), axis=0)


def phase_recon_cl_3d(x, cmap, par):
    fft_dim = par["fft_dim"]
    par["fft_dim"] = (0, 1, 2)
    fft = PyOpenCLnuFFT.create(par["ctx"][0], par["queue"][0], par, par["fft_dim"])

    size = np.shape(x)
    result = np.zeros(
        (par["NScan"], par["NC"], par["NSlice"],
         par["dimY"], par["dimX"]), dtype=DTYPE)
    tmp_result = cla.empty(
        fft.queue,
        (1, par["NC"], par["NSlice"], par["dimY"], par["dimX"]),
        dtype=DTYPE)

    x_shifted = x.copy()

    start = time.time()
    for n in range(size[0]):
        x_shifted[n, ...] = np.fft.fftshift(x[n, ...], axes=(-3, -2, -1))
        clainput = cla.to_device(fft.queue,
                                np.require(
                                    x_shifted[n, ...][None, ...],
                                    requirements='C'))

        fft.FFTH(tmp_result, clainput).wait()

        tmp_result = tmp_result.get()
        tmp_result[n, ...] = np.fft.fftshift(tmp_result[n, ...], axes=(-3, -2, -1))
        result[n, ...] = np.squeeze(tmp_result)
    print("FT took %f s" % (time.time() - start))

    par["fft_dim"] = fft_dim
    return np.stack((np.sum(result[0, ...] * np.conj(cmap[0, ...]), axis=0),
            np.sum(result[0, ...] * np.conj(cmap[1, ...]), axis=0)), axis=0)


def sos_recon(ksp, fft3d=True):
    result = np.zeros_like(ksp)
    print("Performing sum of squares recon...")
    start = time.time()
    for n in range(np.shape(ksp)[0]):
        for c in range(np.shape(ksp)[1]):
            if fft3d:
                # print("Performing 3D FFT...")
                # np.sqrt(np.prod(np.shape(ksp[n, c, ...]))) * \
                result[n, c, ...] = \
                    np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(ksp[n, c, ...]), norm='ortho'))
            else:
                # print("Performing 2D FFT...")
                for z in range(np.shape(ksp)[2]):
                    result[n, c, z, ...] = \
                        np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ksp[n, c, z, ...]), norm='ortho'))
    print("Done! ...FT took %f s." % (time.time() - start))
    return np.sqrt(np.squeeze(np.sum(np.abs(result)**2, axis=1)))


def phase_recon(ksp, cmap, fft3d=True):
    result = np.zeros_like(ksp)
    print("Performing phase recon...")
    start = time.time()
    for n in range(np.shape(ksp)[0]):
        for c in range(np.shape(ksp)[1]):
            if fft3d:
                # print("Performing 3D FFT...")
                # np.sqrt(np.prod(np.shape(ksp[c, ...]))) * \
                result[n, c, ...] = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(ksp[n, c, ...]), norm='ortho')) * \
                                    np.conj(cmap[0, c, ...])
            else:
                # print("Performing 2D FFT...")
                for z in range(np.shape(ksp)[2]):
                    result[n, c, z, ...] = np.fft.ifftshift(np.fft.ifft2(
                        np.fft.fftshift(ksp[n, c, z, ...]), norm='ortho')) * np.conj(cmap[0, c, z, ...])
    print("Done! ...FT took %f s" % (time.time() - start))
    return np.squeeze(np.sum(result, axis=1))


def soft_sense_recon_cl(myargs, par, ksp, cmaps):
    ksp = ksp.astype(DTYPE)
    cmaps = cmaps.astype(DTYPE)

    if myargs.streamed:
        file = resource_filename(
            'pyqmri', 'kernels/OpenCL_Kernels_streamed.c')

        prg = []
        for j in range(len(par["num_dev"])):
            with open(file) as myfile:
                prg.append(Program(
                    par["ctx"][j],
                    myfile.read()))

        op = pyqmirop.OperatorSoftSenseStreamed(par, prg, trafo=False)
        inp_adj = np.require(ksp, requirements='C')
        inp_cmaps = np.require(cmaps, requirements='C')

        return op.adjoop([[inp_adj, inp_cmaps]])
    else:
        file = open(resource_filename('pyqmri', 'kernels/OpenCL_Kernels.c'))

        prg = Program(
            par["ctx"][0],
            file.read())
        file.close()

        op = pyqmirop.OperatorSoftSense(par, prg)

        inp_adj = cla.to_device(op.queue, np.require(ksp, requirements='C'))
        inp_cmaps = cla.to_device(op.queue, np.require(cmaps, requirements='C'))

        return op.adjoop([inp_adj, inp_cmaps]).get()


def _operator_adj_np(ksp, cmaps, mask):
    M, C, Z, Y, X = np.shape(cmaps)
    result = np.zeros_like(cmaps)
    ksp *= mask

    for m in range(M):
        for c in range(C):
            for z in range(Z):
                result[m, c, z, ...] = np.fft.fftshift(
                    np.fft.ifft2(
                        np.fft.ifftshift(ksp[0, c, z, ...], axes=(-2, -1)
                                         ), norm='ortho')
                )
                result[m, c, z, ...] *= np.conj(cmaps[m, c, z, ...])
    return np.sum(result, axis=1)


def _operator_fwd_np(img, cmaps, mask):
    M, C, Z, Y, X = np.shape(cmaps)
    result = np.zeros((1, C, Z, Y, X), dtype=DTYPE)
    for z in range(Z):
        for c in range(C):
            for m in range(M):
                result[0, c, z, ...] += img[m, z, ...] * cmaps[m, c, z, ...]
            result[0, c, z, ...] = np.fft.ifftshift(
                np.fft.fft2(
                    np.fft.fftshift(result[0, c, z, ...], axes=(-2, -1)
                                    ), norm='ortho')
            )
    return mask * result


def _divergence(f):
    df_dx = np.zeros(np.shape(f)[0:-1], dtype=DTYPE)
    df_dy = np.zeros_like(df_dx)
    df_dz = np.zeros_like(df_dx)

    df_dx[..., :-1] = np.diff(f[..., 0], axis=-1)
    df_dy[..., :-1, :] = np.diff(f[..., 1], axis=-2)
    df_dz[:, :-1, ...] = np.diff(f[..., 2], axis=-3)

    return df_dx + df_dy + df_dz


def _gradient(x):
    gradx = np.zeros_like(x)
    grady = np.zeros_like(x)
    gradz = np.zeros_like(x)

    gradx[..., :-1] = np.diff(x, axis=-1)
    grady[..., :-1, :] = np.diff(x, axis=-2)
    gradz[:, :-1, ...] = np.diff(x, axis=-3)

    return np.stack((gradx,
                     grady,
                     gradz), axis=-1)


def _sym_grad(f):
    dfx_dx = np.zeros(np.shape(f)[0:-1], dtype=DTYPE)
    dfy_dy = np.zeros_like(dfx_dx)
    dfz_dz = np.zeros_like(dfx_dx)

    dfy_dx = np.zeros_like(dfx_dx)
    dfz_dx = np.zeros_like(dfx_dx)
    dfz_dy = np.zeros_like(dfx_dx)
    dfx_dy = np.zeros_like(dfx_dx)
    dfx_dz = np.zeros_like(dfx_dx)
    dfy_dz = np.zeros_like(dfx_dx)

    dfx_dx[..., :-1] = np.diff(f[..., 0], axis=-1)
    dfy_dy[..., :-1, :] = np.diff(f[..., 1], axis=-2)
    dfz_dz[:, :-1, ...] = np.diff(f[..., 2], axis=-3)

    dfy_dx[..., :-1] = np.diff(f[..., 1], axis=-1)
    dfz_dx[..., :-1] = np.diff(f[..., 2], axis=-1)
    dfz_dy[..., :-1, :] = np.diff(f[..., 2], axis=-2)

    dfx_dy[..., :-1, :] = np.diff(f[..., 0], axis=-2)
    dfx_dz[..., :-1, :, :] = np.diff(f[..., 0], axis=-3)
    dfy_dz[..., :-1, :, :] = np.diff(f[..., 1], axis=-3)

    return np.stack(
        (
            dfx_dx,
            dfy_dy,
            dfz_dz,
            (dfy_dx + dfx_dy) / 2,
            (dfz_dx + dfx_dz) / 2,
            (dfz_dy + dfy_dz) / 2
        ), axis=-1
    )

def _proximal_op(xi, sigma):
    return xi / (1 + sigma)


def _proximal_op(xi, sigma, lamda):
    return xi / (1 + sigma / lamda)


def _proximal_op_max(xi):
    return xi / np.maximum(1.0, np.linalg.norm(np.abs(xi), axis=-1, keepdims=True))


def _primal_dual_solver_np(ksp, cmaps):
    iters = 100
    yn = np.zeros_like(ksp)
    yn1 = np.empty_like(yn)
    xn = np.zeros((2, 4, 64, 64), dtype=DTYPE)
    xn1 = np.empty_like(xn)
    dx = ksp.copy()
    sigma = np.float32(1 / np.sqrt(12))
    tau = np.float32(1 / np.sqrt(12))

    mask = np.zeros(np.shape(ksp), dtype=DTYPE_real)
    mask[..., ::2, :] = 1

    for i in range(iters):
        if i % 10 == 0:
            print(i)
        yn1 = _proximal_op(yn + sigma * (_operator_fwd_np(xn, cmaps, mask) - dx), sigma, 1)
        xn1 = xn - tau * _operator_adj_np(yn1, cmaps, mask)
        xn1_ = 2 * xn1 - xn
        yn = yn1.copy()
        xn = xn1_.copy()

    img_montage(np.abs(xn), 'PD numpy recon')


# def _primal_dual_solver_tv_np(ksp, cmaps):
#     iters = 100
#     yn = np.zeros_like(ksp)
#     yn1 = np.empty_like(yn)
#     xn = np.zeros((2, 4, 64, 64), dtype=DTYPE)
#     xn1 = np.empty_like(xn)
#     dx = ksp.copy()
#     pn = np.zeros(xn.shape+(3,), dtype=DTYPE)
#     # qn = np.zeros_like(pn)
#     pn1 = np.zeros_like(pn)
#     # qn1 = np.zeros_like(qn)
#
#     sigma = np.float32(1 / np.sqrt(12))
#     tau = np.float32(1 / np.sqrt(12))
#     lamda = 1
#
#     mask = np.zeros(np.shape(ksp), dtype=DTYPE_real)
#     mask[..., ::2, :] = 1
#     dx *= mask
#
#     for i in range(iters):
#         yn1 = _proximal_op(yn + sigma * (_operator_fwd_np(xn, cmaps, mask) - dx), sigma, lamda)
#         grad_x = _gradient(xn)
#         grad = np.moveaxis(grad_x[0], -1, 0)
#
#         pn1 = _proximal_op_max(pn + sigma * grad_x)
#         # qn1 = _proximal_op_max(qn + sigma * grad_x)
#         div_p = _divergence(pn1)
#
#         xn1 = xn - tau * (_operator_adj_np(yn1, cmaps, mask) - div_p)
#         xn1_ = 2 * xn1 - xn
#
#         if i % 10 == 0 or i < 10:
#             print(i)
#             img_montage(np.abs(grad), 'Numpy Gradients')
#             img_montage(np.abs(div_p[0]), 'Numpy Divergence')
#             img_montage(np.abs(xn1_[0]), 'X')
#.flatten()
#         yn = yn1.copy()
#         xn = xn1_.copy()
#         pn = pn1.copy()
#
#     img_montage(np.real(xn), 'PD numpy recon')

def calculate_cost(myargs, par, x, v, ksp, cmaps):
    if myargs.streamed:
        x = np.transpose(x, (1, 0, 2, 3))
        v = np.transpose(v, (1, 0, 2, 3, 4))
        ksp = np.transpose(ksp, (1, 2, 0, 3, 4))
        cmaps = np.transpose(cmaps, (1, 2, 0, 3, 4))

    mask = create_mask(np.shape(ksp), myargs.acceleration_factor, myargs.dim_us)
    out_fwd = _operator_fwd_np(x, cmaps, mask)
    cost = np.linalg.norm(out_fwd - ksp)**2
    # cost = np.vdot((out_fwd - ksp), (out_fwd - ksp)).real

    reg_cost = 0
    if myargs.reg_type == 'TV':
        cost *= myargs.lamda * 0.5
        reg_cost = np.sum(np.abs(_gradient(x)))
    if myargs.reg_type == 'TGV':
        cost *= myargs.lamda * 0.5
        reg_cost = par['alpha1'] * np.sum(np.abs(_gradient(x) - v)) + par['alpha0'] * np.sum(np.abs(_sym_grad(v)))

    return cost + reg_cost