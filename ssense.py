import argparse
import math
import csv
import cv2

from pkg_resources import resource_filename
from pyqmri.pyqmri import _setupOCL
from pyqmri._helper_fun import CLProgram as Program
import pyqmri._helper_fun._utils as utils

from pyqmri._my_helper_fun.import_data import import_data
from pyqmri._my_helper_fun.display_data import img_montage
from pyqmri._my_helper_fun.export_data import *
from pyqmri._my_helper_fun.recon import *
from pyqmri._my_helper_fun.helpers import *

from pyqmri.pdsose import SoftSenseOptimizer
import pyqmri.solver as pyqmrisl


def _set_output_dir(par, myargs):
    if myargs.outdir == '':
        outdir = Path.cwd()
    else:
        outdir = Path(myargs.outdir)
    out_folder = "SoftSense_out"
    if myargs.use_phantom:
        out_folder = "SoftSense_out_phantom"
    if myargs.use_in_vivo:
        out_folder = "SoftSense_out_invivo"
    outdir = outdir / out_folder / myargs.reg_type / time.strftime("%Y-%m-%d_%H")  # -%M-%S")
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    par["outdir"] = outdir


def _setup_par(par, myargs, ksp_data, cmaps):
    ksp_shape = np.shape(ksp_data)
    cmaps_shape = np.shape(cmaps)

    if args.double_precision:
        par["DTYPE"] = np.complex128
        par["DTYPE_real"] = np.float64
    else:
        par["DTYPE"] = np.complex64
        par["DTYPE_real"] = np.float32

    par["C"] = np.require(cmaps, requirements='C').astype(par["DTYPE"])

    par["NScan"] = ksp_shape[0]
    par["dimX"] = ksp_shape[-1]
    par["dimY"] = ksp_shape[-2]
    par["NSlice"] = cmaps_shape[2]
    par["NC"] = cmaps_shape[1]
    par["NMaps"] = cmaps_shape[0]

    par["N"] = par["dimX"]
    par["Nproj"] = par["dimY"]

    par["unknowns_TGV"] = par["NMaps"]
    par["unknowns"] = par["NMaps"]
    par["weights"] = np.ones(par["unknowns"], dtype=par["DTYPE_real"])
    par["dz"] = 1
    # not relevant for my case but necessary for Operator class
    par["unknowns_H1"] = 0

    par["fft_dim"] = (-2, -1)

    par["mask"] = np.require(np.ones((par["dimY"], par["dimX"]), dtype=par["DTYPE_real"]), requirements='C')

    par["R"] = myargs.acceleration_factor

    par["overlap"] = 0
    par["par_slices"] = par["NSlice"]

    if myargs.streamed:
        # par["mask"] = np.require(np.ones(ksp_shape, dtype=par["DTYPE_real"]), requirements='C')
        # par NSlice but x fully sampled
        if myargs.reco_slices == -1:
            par["par_slices"] = int(par["dimX"] / (2 * len(par["num_dev"])))
        else:
            par["par_slices"] = int(myargs.reco_slices / (2 * len(par["num_dev"])))
        par["overlap"] = 1

    _set_output_dir(par, myargs)


def _setup_ss_par(ss_par, myargs):
    ss_par["display_iterations"] = True
    ss_par["adaptivestepsize"] = myargs.adapt_stepsize
    ss_par["tol"] = 1e-8
    ss_par["stag"] = 1e10
    ss_par["sigma"] = np.float32(1 / np.sqrt(12))
    ss_par["lambd"] = myargs.lamda
    ss_par["alpha0"] = np.sqrt(2) if myargs.recon_type == '2D' else np.sqrt(3)
    ss_par["alpha1"] = 1   # 1


def _gen_data_from_imgs(imgs, cmaps, par, type='2D'):
    result = np.zeros(np.shape(cmaps)).astype(np.complex64)
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


def _start_recon(data, par, myargs):
    optimizer = SoftSenseOptimizer(par,
                                   myargs,
                                   myargs.reg_type,
                                   streamed=myargs.streamed,
                                   DTYPE=par["DTYPE"],
                                   DTYPE_real=par["DTYPE_real"])
    result, i = optimizer.execute(data.copy())

    elapsed_time = optimizer._elapsed_time

    del optimizer
    return result, elapsed_time, i


def _cvt_structured_to_complex(data):
    return (data['real'] + 1j * data['imag']).astype(np.complex64)


def _check_data_shape(myargs, ksp_data):
    gpyfft_primes = [2, 3, 5, 7]
    z, y, x = np.shape(ksp_data)[-3:]
    reco_slices = myargs.reco_slices
    while reco_slices > 0:
        data_prime_factors = utils.prime_factors(reco_slices*y*x)
        l = [i for i in data_prime_factors if i not in gpyfft_primes]
        if not l:
            break
        else:
            reco_slices -= 1

    if myargs.reco_slices != reco_slices:
        print("Reducing slices to %i" %reco_slices)

    myargs.reco_slices = reco_slices


def _get_data(myargs):
    # import kspace and coil sensitivity data
    ksp_data = import_data(myargs.kspfile, 'k-space')[0].astype(np.complex64)
    print('k_space input shape (x, y, z, ncoils): ' + str(np.shape(ksp_data)))

    cs_data = import_data(myargs.csfile, 'cmap')[0]
    # cmaps = cs_data.view(DTYPE)
    cmaps = _cvt_structured_to_complex(cs_data)
    print('c_maps input shape (nmaps, ncoils, z, y, x) ' + str(cs_data.shape))

    # reorder kspace to (NScan, NC, NSlice, Ny, Nx)
    ksp_data = np.moveaxis(ksp_data, [0, 1, 2, 3], [3, 2, 1, 0])
    ksp_data = np.expand_dims(ksp_data, axis=0) * 2e4
    ksp_data = prepare_data(ksp_data, rescale=False, recon_type='3D') * 2e4
    ksp_data = np.fft.ifft(ksp_data, axis=-3, norm='ortho')


    myargs.full_dimXY = True
    # x = phase_recon(np.fft.fftshift(ksp_data, axes=(-2, -1)), cmaps, recon_type='2D')
    x = phase_recon(ksp_data, cmaps, recon_type='2D')

    img_montage(sqrt_sum_of_squares(x))


    n_slices = x.shape[1]

    if myargs.reco_slices < 0:
        myargs.reco_slices = n_slices

    _check_data_shape(myargs, ksp_data)

    return ksp_data, cmaps, x


def _normalize_data(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def _add_noise(data, noise_fac):
    return data + noise_fac * (np.random.randn(*np.shape(data)) + 1j * np.random.randn(*np.shape(data)))


def _preprocess_data(myargs, par, kspace, x, full_dim=False):
    swapaxis = -3
    kspace = prepare_data(kspace, rescale=False, recon_type=myargs.type)
    # par["mask"] = np.fft.fftshift(par["mask"], axes=par["fft_dim"])

    if myargs.type == '3D':
        if not full_dim:
            # full_dimY = (np.all(np.abs(kspace[0, 0, 0, :, 0])) or
            #              np.all(np.abs(kspace[0, 0, 0, :, 1])))
            # full_dimX = (np.all(np.abs(kspace[0, 0, 0, 0, :])) or
            #              np.all(np.abs(kspace[0, 0, 0, 1, :])))
            full_dimX = True
            full_dimY = False
            # full_dimY = True

            dimX = par["dimX"]
            dimY = par["dimY"]
            NSlice = par["NSlice"]
            if full_dimX and not full_dimY:
                swapaxis = -1
                par["dimX"] = NSlice
                par["N"] = NSlice
                par["NSlice"] = dimX
            elif full_dimY and not full_dimX:
                swapaxis = -2
                par["dimY"] = NSlice
                par["Nproj"] = NSlice
                par["NSlice"] = dimY
            # par["par_slices"] = par["NSlice"]
            kspace = np.fft.ifft(kspace, axis=swapaxis, norm='ortho')
            kspace = np.require(
                np.swapaxes(kspace, swapaxis, -3),
                requirements='C')
            par["C"] = np.require(
                np.swapaxes(par["C"], swapaxis, -3),
                requirements='C')
            if par["mask"].ndim > 2:
                par["mask"] = np.require(
                    np.swapaxes(par["mask"], swapaxis, -3),
                    requirements='C')
                img_montage(par["mask"][0, 0, [0]], 'Subsampling mask')
                par["mask"] = np.fft.fftshift(par["mask"], axes=par["fft_dim"])
                # par["mask"] = par["mask"][0, 0, 0, ...]

    imgs = phase_recon(kspace, par["C"], recon_type='2D')
    par["max_val"] = np.max(sqrt_sum_of_squares(imgs))

    n_slices = np.shape(imgs)[abs(swapaxis)]
    # n_slices = 180

    n_par_slices = myargs.reco_slices if myargs.reco_slices > 0 else n_slices
    if n_par_slices != n_slices and n_par_slices > 0:
        par["NSlice"] = n_par_slices
        if not myargs.streamed:
            par["par_slices"] = n_par_slices

        slice_idx = (int(n_slices / 2) - int(math.floor(n_par_slices / 2)),
                     int(n_slices / 2) + int(math.ceil(n_par_slices / 2)))
        x = np.swapaxes(x, swapaxis, -3)
        # imgs = np.swapaxes(imgs, swapaxis, -3)
        x = np.require(x[:, slice_idx[0]:slice_idx[-1], :, :].copy(),
                       requirements='C').astype(par["DTYPE"])
        imgs = np.require(imgs[:, slice_idx[0]:slice_idx[-1], :, :].copy(),
                          requirements='C').astype(par["DTYPE"])
        kspace = np.require(kspace[..., slice_idx[0]:slice_idx[-1], :, :],
                            requirements='C').astype(par["DTYPE"])
        par["C"] = np.require(par["C"][..., slice_idx[0]:slice_idx[-1], :, :],
                              requirements='C').astype(par["DTYPE"])
        if par["mask"].ndim > 2:
            par["mask"] = np.require(par["mask"][..., slice_idx[0]:slice_idx[-1], :, :],
                                     requirements='C').astype(par["DTYPE_real"])

    # img_montage(sqrt_sum_of_squares(imgs), 'Selected phase sensitive recon images')

    return kspace, x


def _get_phantom_data(myargs):
    with h5py.File(myargs.datafile, 'r') as f:
        img_ref1 = np.array(f.get('data/M0_img_red/ref1'))
        img_ref2 = np.array(f.get('data/M0_img_red/ref2'))
        kspace = np.array(f.get('data/M0_kspace'))
        cmap1 = np.array(f.get('data/coils_red/b1'))
        cmap2 = np.array(f.get('data/coils_red/b2'))
    f.close()

    kspace = _cvt_structured_to_complex(kspace).copy()
    cmap1 = _cvt_structured_to_complex(cmap1).copy()
    cmap2 = _cvt_structured_to_complex(cmap2).copy()
    img_ref1 = _cvt_structured_to_complex(img_ref1).copy()
    img_ref2 = _cvt_structured_to_complex(img_ref2).copy()

    kspace = np.require(np.expand_dims(kspace, axis=0).astype(np.complex64), requirements='C')
    kspace_noisy = _add_noise(kspace.copy(), noise_fac=myargs.noise).astype(DTYPE)
    # kspace_noisy = kspace.copy()

    cmaps = np.require(np.stack((cmap1, cmap2), axis=0), requirements='C')
    imgs = np.require(np.stack((img_ref1, img_ref2), axis=0), requirements='C')
    imgs = _add_noise(imgs.copy(), noise_fac=myargs.noise).astype(DTYPE)

    kspace_noisy = np.swapaxes(kspace_noisy, -1, -2)
    imgs = np.swapaxes(imgs, -1, -2)
    cmaps = np.swapaxes(cmaps, -1, -2)

    return kspace_noisy, cmaps, imgs


def _read_data(file_name):
    with h5py.File(file_name, 'r') as f:
        data = np.array(f.get(list(f.keys())[0]))
    f.close()
    return data


def _get_in_vivo_data(myargs, data='data2', sampling='C2x2s0'):
    data_acl_file = Path.cwd() / 'Granat' / 'Results' / sampling / 'data_ACL.mat'
    #data_kspace_file = Path.cwd() / 'Granat2' / 'Results' / sampling / 'kspace_sc_cc.mat'
    cmaps_file = Path.cwd() / 'Granat' / 'Results' / sampling / 'sensitivities_ecalib.mat'
    mask_file = Path.cwd() / 'Granat' / 'Results' / sampling / 'mask_{}.mat'.format(sampling)

    data_full_file = Path.cwd() / 'Granat' / 'Results' / 'full' / 'kspace_sc_cc.mat'
    cmaps_full_file = Path.cwd() / 'Granat' / 'Results' / 'full' / 'sensitivities_full_ecalib_coilcomp_1.mat'

    # img_file = Path.cwd() / 'Granat' / 'Results' / sampling / 'img_sos.mat'
    # img_x = _read_data(img_file)
    # img_montage(img_x)
    # img_montage(np.swapaxes(img_x, -1, -3))

    data_acl = _cvt_structured_to_complex(_read_data(data_acl_file))
    # data_kspace = _cvt_structured_to_complex(_read_data(data_k_space_file))
    cmaps = _cvt_structured_to_complex(_read_data(cmaps_file))
    data_full = _cvt_structured_to_complex(_read_data(data_full_file))
    cmaps_full = _cvt_structured_to_complex(_read_data(cmaps_full_file))

    mask = np.expand_dims(_read_data(mask_file), axis=0)

    kspace_data = np.require(np.expand_dims(data_acl, axis=0).astype(np.complex64), requirements='C') * 1e7
    #kspace_data = prepare_data(kspace_data, rescale=False, recon_type='3D')  # / np.linalg.norm(kspace_data_full)
    # dscale = 1 / np.linalg.norm(kspace_data)    img_montage(sqrt_sum_of_squares(imgs), 'fully sampled images')
    # kspace_data *= dscale
    kspace_data_full = np.require(np.expand_dims(data_full, axis=0).astype(np.complex64), requirements='C') * 1e7
    kspace_data_full = prepare_data(kspace_data_full, rescale=False, recon_type='3D')  # / np.linalg.norm(kspace_data_full)

    cmaps = np.require(cmaps, requirements='C').astype(np.complex64)
    cmaps_full = np.require(cmaps_full, requirements='C').astype(np.complex64)

    imgs = phase_recon(kspace_data_full, cmaps_full)
    # imgs2 = phase_recon(kspace_data, cmaps)
    #
    # img_montage(sqrt_sum_of_squares(imgs))
    # img_montage(sqrt_sum_of_squares(imgs2))

    #cmap1 = cmaps[[0]].copy()
    # mask = mask[0, 0, 0, :, :]
    # img_montage(sqrt_sum_of_squares(np.swapaxes(imgs, -3, -1)))
    # img_montage(sqrt_sum_of_squares(np.swapaxes(imgs, -3, -1))[125:131])
    # img_montage(sqrt_sum_of_squares(np.swapaxes(imgs, -3, -1))[[128]])

    # kspace_data = kspace_data.repeat(2, axis=-3).repeat(2, axis=-2)
    # cmaps = cmaps.repeat(2, axis=-3).repeat(2, axis=-2)
    # mask = mask.repeat(2, axis=-3).repeat(2, axis=-2)

    return kspace_data, cmaps, imgs, mask


def _get_undersampling_mask(par, myargs, file='', R=2):
    if file in ['us_z', 'us_y', 'us_x']:
        mask = np.zeros((par["NSlice"], par["dimY"], par["dimX"]), dtype=par["DTYPE_real"])
        if file == 'us_z':
            mask[::R, ...] = 1
        if file == 'us_y':
            mask[..., ::R, :] = 1
            mask[..., 71:88, :] = 1
        if file == 'us_x':
            mask[..., ::R] = 1
        par["mask"] = mask

    elif myargs.maskfile:
        # with h5py.File(myargs.maskfile, 'r') as f:
        #     mask = np.array(f.get(myargs.maskfile.stem))
        # f.close()
        mask = _read_data(myargs.maskfile)

        mask = np.require(mask.astype(par["DTYPE_real"]), requirements='C')
        par["mask"] = mask[0, ...]

    R = np.size(par["mask"]) / np.sum(par["mask"])
    myargs.acceleration_factor = R
    print('Acceleration factor is: {:.2f}'.format(R))

    par["mask"] = np.reshape(
        np.tile(
            par["mask"],
            (par["NScan"] * par["NC"], 1, 1)),
        (par["NScan"], par["NC"], par["NSlice"], par["dimY"], par["dimX"])
    )

    # par["mask"] = np.fft.fftshift(par["mask"], axes=(-3, -2, -1))


def _write_to_csv_file_lambda(par, myargs, noise_fac, metrics, el_time):
    with open(par["outdir"] / "metrics.csv", "a", newline='') as file:
        fieldnames = ['reg', 'noise', 'lambda', 'MSE_mean',
                      'PSNR_mean', 'PSNR_std', 'PSNR_max', 'PSNR_min',
                      'SSIM_mean', 'SSIM_std', 'SSIM_max', 'SSIM_min',
                      'Elapsed time']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writerow({'reg': str(myargs.reg_type),
                         'noise': str(noise_fac),
                         'lambda': str(myargs.lamda),
                         'MSE_mean': str(np.mean(metrics[0])),
                         'PSNR_mean': str(np.mean(metrics[1])),
                         'PSNR_std': str(np.std(metrics[1])),
                         'PSNR_max': str(np.max(metrics[1])),
                         'PSNR_min': str(np.min(metrics[1])),
                         'SSIM_mean': str(np.mean(metrics[2])),
                         'SSIM_std': str(np.std(metrics[2])),
                         'SSIM_max': str(np.max(metrics[2])),
                         'SSIM_min': str(np.min(metrics[2])),
                         'Elapsed time': str(el_time)
                         })


def _write_to_csv_file_masks(par, myargs, acc, mask, lamda, noise_fac, metrics, el_time, i):
    with open(par["outdir"] / "phantom_eval.csv", "a", newline='') as file:
        fieldnames = ['R', 'mask', 'noise', 'lambda', 'MSE_mean',
                      'PSNR_mean', 'PSNR_std', 'PSNR_max', 'PSNR_min',
                      'SSIM_mean', 'SSIM_std', 'SSIM_max', 'SSIM_min',
                      'Elapsed time', 'Iterations']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writerow({'R': str(acc),
                         'mask': mask,
                         'noise': str(noise_fac),
                         'lambda': str(myargs.lamda),
                         'MSE_mean': str(np.mean(metrics[0])),
                         'PSNR_mean': str(np.mean(metrics[1])),
                         'PSNR_std': str(np.std(metrics[1])),
                         'PSNR_max': str(np.max(metrics[1])),
                         'PSNR_min': str(np.min(metrics[1])),
                         'SSIM_mean': str(np.mean(metrics[2])),
                         'SSIM_std': str(np.std(metrics[2])),
                         'SSIM_max': str(np.max(metrics[2])),
                         'SSIM_min': str(np.min(metrics[2])),
                         'Elapsed time': str(el_time),
                         'Iterations': str(i)
                         })


def _eval_mask(par, myargs, ksp_data, cmaps, x):

    # x_noisy = _add_noise(x.copy(), myargs.noise).astype(par["DTYPE"])
    max_val = np.max(sqrt_sum_of_squares(x))
    par["max_val"] = max_val

    for filedir in ['R9']:  # ['R8']
        R = int(filedir.split('R')[-1])
        myargs.acceleration_factor = R
        par["R"] = myargs.acceleration_factor
        for file in ['mask_CAIPI4x4x2.mat', 'mask_a.mat']:

            _setup_par(par, myargs, ksp_data, cmaps)
            ksp_data_ = ksp_data.copy()

            myargs.maskfile = Path.cwd() / 'data_soft_sense_test' / 'us_masks' / filedir / file
            myargs._mask_file = file

            _get_undersampling_mask(par, myargs, file, R)

            ksp_data_ = ksp_data_ * par["mask"] if par["mask"].ndim > 2 else ksp_data_
            ksp_data_, x = _preprocess_data(myargs, par, ksp_data_, x, myargs.full_dimXY)

            par["mask"] = par["mask"][0, 0, 0, ...]
            par["mask"] = np.require(par["mask"], requirements='C', dtype=par["DTYPE_real"])
            ksp_data_ = ksp_data_.astype(par["DTYPE"])

            # ksp_data_ = np.swapaxes(ksp_data_, -3, -2)
            # par["C"] = np.swapaxes(par["C"], -3, -2)
            x_ssense, recon_time, i = _start_recon(ksp_data_, par, myargs)

            if myargs.reco_slices != -1 and myargs.streamed:
                x_ssense = np.swapaxes(x_ssense, -3, -1)

            x_ssense_ = normalize_imgs(sqrt_sum_of_squares(x_ssense)).astype(par["DTYPE_real"])
            x_ = normalize_imgs(sqrt_sum_of_squares(x)).astype(par["DTYPE_real"])
            # x_diff = x_ssense_ - x_

            # img_montage(sqrt_sum_of_squares(x_ssense), 'Softsense recon ' + myargs.reg_type)
            img_montage(sqrt_sum_of_squares(x_ssense)[[int(np.shape(x_ssense)[-3]/2)], ...])
            # img_montage(x_diff, 'Diffs')
            # x_noisy = np.swapaxes(x_noisy, -3, -1).astype(par["DTYPE"])

            m_fac = 2
            m_size = int(160 * m_fac)
            m_cen = int(m_size / 2)
            m_win = int(m_size / 4)
            m_l, m_t = m_cen - m_win, m_cen + m_win

            x_1 = np.squeeze(sqrt_sum_of_squares(x_ssense.copy())[[int(np.shape(x_ssense)[-3]/2)], ...])

            x_small_1 = cv2.resize(x_1, (m_size, m_size), interpolation=cv2.INTER_NEAREST)[m_l:m_t, m_l:m_t]
            x_small_2 = cv2.resize(x_1, (m_size, m_size), interpolation=cv2.INTER_NEAREST)[0:m_cen, 0:m_cen]

            # _show_img(x_1, max_val, 0)
            try:
                save_img(x_1, args.reg_type + '_res_x_{}'.format(file.split('.')[0]), max_val, 0)
                save_img(x_small_1, args.reg_type + '_res_x_small_1_{}'.format(file.split('.')[0]), max_val, 0)
                save_img(x_small_2, args.reg_type + '_res_x_small_2_{}'.format(file.split('.')[0]), max_val, 0)
            except Exception as ex:
                print('something whent wrong...')

            metrics = calc_image_metrics(x_ssense_[[int(np.shape(x_ssense_)[-3]/2)], ...], x_[[int(np.shape(x_)[-3]/2)], ...])
            # x_noisy = normalize_imgs(sqrt_sum_of_squares(x_noisy.copy())).astype(par["DTYPE_real"])
            # metrics_ = calc_image_metrics(x_ssense_, x_noisy)
            _write_to_csv_file_masks(par, myargs, myargs.acceleration_factor, myargs._mask_file, myargs.lamda, myargs.noise, metrics, recon_time, i)
            # _write_to_csv_file_masks(par, myargs, myargs.acceleration_factor, myargs._mask_file, myargs.lamda, myargs.noise, metrics_, recon_time)


def _eval_in_vivo(par, myargs, ksp_data, x, mask, mask_type='C2x2s0.mat'):
    par["mask"] = np.require(mask, requirements='C', dtype=par["DTYPE_real"])

    max_val_ref = np.max(sqrt_sum_of_squares(x))
    # par["max_val"] = max_val

    ksp_data_, x = _preprocess_data(myargs, par, ksp_data, x, myargs.full_dimXY)
    ksp_data_ = ksp_data_.astype(par["DTYPE"])
    par["mask"] = par["mask"][0, 0, 0, ...]

    myargs._mask_file = mask_type

    for lamda in [0.1]:
        myargs.lamda = lamda

        x_ssense, recon_time, i = _start_recon(ksp_data_, par, myargs)

        if myargs.reco_slices != -1 and myargs.streamed:
            x_ssense = np.swapaxes(x_ssense, -3, -1)

        #x_ssense_ = normalize_imgs(sqrt_sum_of_squares(x_ssense)).astype(par["DTYPE_real"])
        #x_ = normalize_imgs(sqrt_sum_of_squares(x)).astype(par["DTYPE_real"])
        # x_diff = x_ssense_ - x_

        #img_montage(sqrt_sum_of_squares(x_ssense), 'Softsense recon ' + myargs.reg_type)
        #img_montage(sqrt_sum_of_squares(x_ssense)[[int(np.shape(x_ssense)[-3] / 2)], ...])
        # img_montage(x_diff, 'Diffs')

        # m_fac = 2
        # m_size1 = int(np.shape(x)[-2] * m_fac)
        # m_size2 = int(np.shape(x)[-1] * m_fac)
        # m_cen1 = int(m_size1 / 2)
        # m_cen2 = int(m_size2 / 2)
        # m_win1 = int(m_size1 / 4)
        # m_win2 = int(m_size2 / 4)
        # m_l1, m_t1 = m_cen1 - m_win1, m_cen1 + m_win1
        # m_l2, m_t2 = m_cen2 - m_win2, m_cen2 + m_win2
        #
        # x_1 = np.squeeze(sqrt_sum_of_squares(x_ssense.copy())[[int(np.shape(x_ssense)[-3] / 2)], ...])
        # x_ref = np.squeeze(sqrt_sum_of_squares(x.copy())[[int(np.shape(x)[-3] / 2)], ...])
        #
        # x_small_1 = x_1[7:-7, 7:-7]
        # x_small_2 = cv2.resize(x_small_1, (116, 180), interpolation=cv2.INTER_NEAREST) # [m_l1:m_t1, m_l2:m_t2]
        #
        # x_small_1_ref = x_ref[7:-7, 7:-7]
        # x_small_2_ref = cv2.resize(x_small_1_ref, (116, 180), interpolation=cv2.INTER_NEAREST) # [m_l1:m_t1, m_l2:m_t2]
        #
        # #x_small_2 = cv2.resize(x_1, (m_size2, m_size1), interpolation=cv2.INTER_LINEAR)[15:-15, 15:-15]  # [m_l1:m_t1, m_l2:m_t2]
        #
        # # x_small_2 = cv2.resize(x_1, (m_size, m_size), interpolation=cv2.INTER_NEAREST)[0:m_cen, 0:m_cen]
        # max_val = par["max_val"]
        # _show_img(x_small_2, max_val, 0)
        # _show_img(x_small_2_ref, max_val_ref, 0)
        #
        # try:
        #     save_img(x_1, args.reg_type + '_res_x_{}'.format(mask_type.split('.')[0]) + '_{}'.format(int(lamda*1000)), max_val, 0)
        #     save_img(x_ref, 'ref', max_val_ref, 0)
        #     save_img(x_small_2_ref, 'ref_small', max_val_ref, 0)
        #     # save_img(x_small_1, args.reg_type + '_res_x_small_1_{}'.format(mask_type.split('.')[0]) + '_{}'.format(int(lamda*1000)), max_val, 0)
        #     save_img(x_small_2, args.reg_type + '_res_x_small_2_{}'.format(mask_type.split('.')[0]) + '_{}'.format(int(lamda*1000)), max_val, 0)
        # except Exception as ex:
        #     print('something whent wrong...')

        # _show_img(np.flipud(x_ssense_[36, :, :]), np.max(x_ssense_), 0)
        # _show_img(np.flipud(x_[36, :, :]), np.max(x_), 0)

        #metrics = calc_image_metrics(x_ssense_, x_)
        #_write_to_csv_file_masks(par, myargs, myargs.acceleration_factor, myargs._mask_file, myargs.lamda, myargs.noise,
        #                         metrics, recon_time, i)
        # metrics = calc_image_metrics(normalize_imgs(np.expand_dims(x_small_2, 0)), normalize_imgs(np.expand_dims(x_small_2_ref, 0)))
        # _write_to_csv_file_masks(par, myargs, myargs.acceleration_factor, myargs._mask_file, myargs.lamda, myargs.noise,
        #                          metrics, recon_time, i)


def save_img(img, filename='', max_val=1, min_val=0):
    img_rescaled = (255.0 / max_val * (img - min_val)).astype(np.uint8)
    im = Image.fromarray(img_rescaled)
    directory = '/home/chrgla/masterthesis/deployed/results/imgs/'
    outdir = Path(directory)
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    file_dir = directory + filename + '.png'
    im.save(file_dir)


def _show_img(img, max_val, min_val):
    plt.figure()
    plt.imshow(img, cmap='gray', vmax=max_val, vmin=min_val)
    plt.show()


def _eval_lambda(par, myargs, ksp_data, cmaps, x, mask_file='C2x2s0.mat'):
    myargs._mask_file = mask_file
    for noise_fac in [0]: #, 6.5, 13, 26, 65]:  # for phantom data --> SNR [inf, 100, 50, 25, 10]:
        myargs.noise = noise_fac
        max_val = 1

        x_noisy = _add_noise(x.copy(), noise_fac=myargs.noise).astype(par["DTYPE"])

        if myargs.reco_slices != -1:
            max_val = np.max(sqrt_sum_of_squares(x_noisy))
            par["max_val"] = max_val
            # x_noisy = np.swapaxes(x_noisy, -3, -1).astype(par["DTYPE"])
            n_par_slices = myargs.reco_slices
            n_slices = np.shape(x_noisy)[-3]
            slice_idx = (int(n_slices / 2) - int(math.floor(n_par_slices / 2)),
                         int(n_slices / 2) + int(math.ceil(n_par_slices / 2)))

            x_1 = sqrt_sum_of_squares(x_noisy[:, 36, ...].copy())
            x_noisy = x_noisy[:, slice_idx[0]:slice_idx[-1], ...]
            x_noisy = normalize_imgs(sqrt_sum_of_squares(x_noisy))

            # m_fac = 2
            # m_size = int(160 * m_fac)
            # m_cen = int(m_size / 2)
            # m_win = int(m_size / 4)
            # m_l, m_t = m_cen - m_win, m_cen + m_win
            # x_small = cv2.resize(x_1, (m_size, m_size), interpolation=cv2.INTER_NEAREST)[m_l:m_t, m_l:m_t]
            #
            # x_small_2 = cv2.resize(x_1, (m_size, m_size), interpolation=cv2.INTER_NEAREST)[0:m_cen, 0:m_cen]
            # _show_img(x_1, max_val, 0)
            save_img(x_1, 'ref_n_{}'.format(noise_fac * 100), max_val, 0)
            # save_img(x_small, 'ref_small1_n_{}'.format(noise_fac * 100), max_val, 0)
            # save_img(x_small_2, 'ref_small2_n_{}'.format(noise_fac * 100), max_val, 0)

        # lamda_list = [10, 2, 1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01] if myargs.reg_type != 'NoReg' else [1]
        # lamda_list = [50]

        lamda_list = [100.0, 10, 1.0, 0.1, 0.01] if myargs.reg_type != 'NoReg' else [1]

        for lamda in lamda_list:
            myargs.lamda = lamda

            ksp_data_ = ksp_data.copy()
            _setup_par(par, myargs, ksp_data_, cmaps)

            ksp_data_noise = _add_noise(ksp_data_, noise_fac=myargs.noise).astype(DTYPE)
            # ksp_data_noise = ksp_data_

            #mask_folder = 'R9' if myargs.use_phantom else 'in_vivo'
            #myargs.maskfile = Path.cwd() / 'data_soft_sense_test' / 'us_masks' / mask_folder / mask_file
            mask_file = 'us_y'
            _get_undersampling_mask(par, myargs, mask_file, 6)

            ksp_data_ = ksp_data_noise * par["mask"] if par["mask"].ndim > 2 else ksp_data_noise
            ksp_data_, x_ = _preprocess_data(myargs, par, ksp_data_, x.copy(), myargs.full_dimXY)
            ksp_data_ = ksp_data_.astype(par["DTYPE"])

            par["mask"] = par["mask"][0, 0, 0, ...]
            par["mask"] = np.require(par["mask"], requirements='C', dtype=par["DTYPE_real"])

            x_ssense, elapsed_time, i = _start_recon(ksp_data_, par, myargs)

            x_ssense_ = normalize_imgs(np.squeeze(sqrt_sum_of_squares(x_ssense))).astype(par["DTYPE_real"])
            x_ = normalize_imgs(np.squeeze(sqrt_sum_of_squares(x_))).astype(par["DTYPE_real"])
            x_diff = x_ssense_ - x_

            # img_montage(sqrt_sum_of_squares(x_ssense), 'Softsense recon ' + myargs.reg_type)
            img_montage(np.squeeze(sqrt_sum_of_squares(x_ssense))[[int(np.shape(x_ssense)[-3] / 2)], ...], 'Softsense recon ' + myargs.reg_type)
            #img_montage(x_[[int(np.shape(x_)[-3] / 2)], ...], 'Fully sampled image')
            #img_montage(x_diff[[int(np.shape(x_ssense)[-3] / 2)], ...], 'Diffs')

            metrics = calc_image_metrics(x_ssense_, x_noisy)
            # metrics = calc_image_metrics(x_ssense_, x_)
            _write_to_csv_file_lambda(par, myargs, myargs.noise, metrics, elapsed_time)

            x_1 = sqrt_sum_of_squares(x_ssense[:, int(np.shape(x_ssense)[-3] / 2)].copy())

            m_fac = 2
            m_size = int(160 * m_fac)
            m_cen = int(m_size / 2)
            m_win = int(m_size / 4)
            m_l, m_t = m_cen - m_win, m_cen + m_win
            x_small = cv2.resize(x_1, (m_size, m_size), interpolation=cv2.INTER_NEAREST)[m_l:m_t, m_l:m_t]

            x_small_2 = cv2.resize(x_1, (m_size, m_size), interpolation=cv2.INTER_NEAREST)[0:m_cen, 0:m_cen]
            # _show_img(x_1, max_val, 0)
            save_img(x_1, myargs.reg_type + '_recon_n_{}_l_{}'.format(int(noise_fac * 100), int(lamda*1000)), max_val, 0)
            save_img(x_small, myargs.reg_type + '_recon_small1_n_{}_l_{}'.format(int(noise_fac * 100), int(lamda*1000)), max_val, 0)
            save_img(x_small_2, myargs.reg_type + '_recon_small2_n_{}_l_{}'.format(int(noise_fac * 100), int(lamda*1000)), max_val, 0)


def _main(myargs, data='', masktype=''):

    if myargs.use_phantom:
        ksp_data_, cmaps, x = _get_phantom_data(myargs)
    elif myargs.use_in_vivo:
        ksp_data_, cmaps, x, mask = _get_in_vivo_data(myargs, data, masktype)
    else:
        ksp_data_, cmaps, x = _get_data(myargs)

    # setup PyQMRI parameters and PyOCL
    par = {}
    _setupOCL(myargs, par)
    _setup_par(par, myargs, ksp_data_, cmaps)

    _eval_in_vivo(par, myargs, ksp_data_, x, mask)
    # _eval_lambda(par, myargs, ksp_data_, cmaps, x, 'mask_{}.mat'.format(masktype))
    # _eval_mask(par, myargs, ksp_data_, cmaps, x)


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
      '--recon_type', default='3D', dest='type',
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
    args.add_argument(
        '--adapt_stepsize', default='0', dest='adapt_stepsize', type=_str2bool,
        help='Enable accelerated optimization by adaptive step size computation.')
    args.add_argument(
        '--use_phantom', default='0', dest='use_phantom', type=_str2bool,
        help='Use phantom instead...'
    )
    args.add_argument(
        '--use_in_vivo', default='0', dest='use_in_vivo', type=_str2bool,
        help='Use in-vivo data instead...'
    )

    args = args.parse_args()

    args.trafo = False
    args.use_GPU = True
    args.streamed = False
    args.devices = -1
    args.datafile = Path.cwd() / 'Granat' / 'Results' / 'C2x2s0' / 'data_softSense.mat'
    args.maskfile = Path.cwd() / 'data_soft_sense_test' / 'us_masks' / 'R4' 'mask_CAIPI2x2x0.mat'
    args.kspfile = Path.cwd() / 'data_soft_sense_test' / 'kspace.mat'
    args.csfile = Path.cwd() / 'data_soft_sense_test' / 'sensitivities_ecalib.mat'

    args.double_precision = True

    args.use_phantom = False
    args.use_in_vivo = True

    args.type = '3D'
    # args.reg_type = 'TV'  # 'NoReg', 'TV', or 'TGV'

    args.reco_slices = 64
    args.adapt_stepsize = True
    args.full_dimXY = False
    args.acceleration_factor = 4

    np.random.seed(42)

    args.noise = 0
    for reg_type in ['TV', 'TGV']:
        args.reg_type = reg_type
        #for mask in ['CAIPI2x2x0', 'CAIPI2x2x1', 'a']:  #, 6.5, 13, 26, 65]:
            # args.noise = noise
        args.lamda = 1.0
        _main(args, masktype='C2x2s0')

    # args.noise = 0
    # for data in ['data1']:
    #     for mask in ['C2x2x0', 'C2x2x1']:
    #         for reg_type in ['TV', 'TGV']:
    #             args.lamda = 0.1
    #             args.reg_type = reg_type
    #
    #             _main(args, data, mask)
