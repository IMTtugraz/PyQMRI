from pyqmri._my_helper_fun.recon import *
from pyqmri._my_helper_fun.helpers import *
from pkg_resources import resource_filename
from pyqmri._helper_fun import CLProgram as Program


DTYPE = np.complex64
DTYPE_real = np.float32


def _test_kernel(par):
    file = open(resource_filename('pyqmri', 'kernels/OpenCL_Kernels.c'))
    prg = Program(
        par["ctx"][0],
        file.read())
    file.close()

    queue = par["queue"][0]
    x_in = np.ones((2, 4, 6)) + 1j * np.ones((2, 4, 6))
    x_in = x_in.astype(DTYPE) * 0.6
    Ky_in = (np.ones_like(x_in) + 1j * np.ones_like(x_in)) * 0.1
    x_out = np.empty_like(x_in)
    x_in = cla.to_device(queue, x_in)
    Ky_in = cla.to_device(queue, Ky_in)
    x_out = cla.to_device(queue, x_out)

    tau = 1.0

    prg.update_x_1d(
        queue, (x_in.size,), None,
        x_out.data,
        x_in.data,
        Ky_in.data,
        np.float32(tau),
        np.float32(0),
        wait_for=x_out.events+x_in.events+Ky_in.events).wait()
    res = x_out.get()


def _test_operator(ksp, cmaps, par):
    imgs = phase_recon_cl_3d(ksp, cmaps, par)
    img_montage(np.real(np.squeeze(imgs)), 'Phase sensitive reconstruction 3D')
    # img_montage(np.imag(np.squeeze(imgs)), 'Phase sensitive reconstruction 3D')
    # img_montage(np.abs(np.squeeze(imgs)), 'Phase sensitive reconstruction 3D')

    img_montage(np.abs(ksp[:, 25, 23:27, ...]), 'ksp original')

    ksp_data2d = gen_2ddata_from_imgs(imgs, cmaps)
    par["fft_dim"] = (1, 2)

    # Select only several slices
    NSlice = 4
    par["NSlice"] = NSlice
    ksp_data2d = ksp_data2d[:, :, 21:21+NSlice, :, :]
    cmaps = cmaps[:, :, 21:21+NSlice, ...]
    imgs_sel = imgs[:, 21:21+NSlice, ...]
    # img_montage(np.abs(np.squeeze(imgs)), 'Original selected images')

    img_montage(np.abs(ksp_data2d[:, 25, ...]), 'ksp 2d')

    file = open(resource_filename('pyqmri', 'kernels/OpenCL_Kernels.c'))
    prg = Program(
        par["ctx"][0],
        file.read())
    file.close()

    op = pyqmirop.OperatorSoftSense(par, prg)

    # ksp_data2d = np.fft.ifftshift(ksp_data2d, axes=(-2, -1))
    # img_montage(np.abs(ksp_data2d[:, 25, ...]), 'ksp 2d shifted')
    ksp = ksp_data2d.astype(DTYPE)
    inp_adj = cla.to_device(op.queue, np.require(ksp, requirements='C'))
    cmaps = cmaps.astype(DTYPE)
    inp_cmaps = cla.to_device(op.queue, np.require(cmaps, requirements='C'))

    out_adj = op.adjoop([inp_adj, inp_cmaps]).get()
    img_montage(np.real(out_adj), 'Out adjoint real')
    img_montage(np.imag(out_adj), 'Out adjoint imag')
    # img_montage(np.abs(imgs), 'Original')

    out_adj_np = _operator_adj_np(ksp, cmaps)
    print(np.allclose(out_adj, out_adj_np))

    out_adj = out_adj.astype(DTYPE)
    inp_fwd = cla.to_device(op.queue, np.require(out_adj, requirements='C'))
    cmaps = cmaps.astype(DTYPE)
    inp_cmaps = cla.to_device(op.queue, np.require(cmaps, requirements='C'))

    out_fwd = op.fwdoop([inp_fwd, inp_cmaps]).get()
    # out = np.fft.fftshift(out, axes=(-2, -1))
    img_montage(np.abs(out_fwd[:, 25, ...]), 'ksp forward output')

    out_fwd_np = _operator_fwd_np(out_adj_np, cmaps)
    img_montage(np.abs(out_fwd_np[:, 25, ...]), 'Out fwd np')
    print(np.allclose(out_fwd, out_fwd_np, atol=1e-06))

    # grad_op = pyqmirop.OperatorFiniteGradient(par, prg)
    #
    # inp_fwd = cla.to_device(grad_op.queue, np.require(imgs, requirements='C'))
    # grad_imgs = grad_op.fwdoop(inp_fwd).get()
    # grad_imgs = grad_imgs.astype(DTYPE)
    # grad_imgs_ro = np.moveaxis(grad_imgs[0], -1, 0)[:-1]
    # img_montage(np.abs(grad_imgs_ro), 'Gradient images')
    # # img_montage(np.real(grad_imgs_ro), 'Gradient images')
    #
    # gradx = np.zeros_like(imgs)
    # grady = np.zeros_like(imgs)
    # gradz = np.zeros_like(imgs)
    #
    # gradx[..., :-1] = np.diff(imgs, axis=-1)
    # grady[..., :-1, :] = np.diff(imgs, axis=-2)
    # gradz[:, :-1, ...] = np.diff(imgs, axis=-3)
    #
    # grad = np.stack((gradx,
    #                  grady,
    #                  gradz), axis=-1)
    # div = np.sum(grad, axis=-1)
    # # img_montage(np.abs(div), 'Numpy Divergence')
    # grad = np.moveaxis(grad[0], -1, 0)
    # # img_montage(np.abs(grad), 'Numpy Gradients')