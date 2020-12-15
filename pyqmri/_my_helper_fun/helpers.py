import numpy as np

from skimage.metrics import structural_similarity as calc_ssim

DTYPE = np.complex64
DTYPE_real = np.float32


def gen_2ddata_from_imgs(imgs, cmaps):
    result = np.zeros(np.shape(cmaps)).astype(DTYPE)
    for z in range(np.shape(imgs)[1]):
        for c in range(np.shape(cmaps)[1]):
            for m in range(np.shape(cmaps)[0]):
                # result[m, c, z, ...] = 1 * np.fft.ifftshift(np.fft.fft2(
                #     np.fft.fftshift(imgs[m, z, ...] * cmaps[m, c, z, ...]), norm='ortho'))
                result[m, c, z, ...] = np.fft.fft2(imgs[m, z, ...] * cmaps[m, c, z, ...], norm='ortho')
    return np.sum(result, axis=0, keepdims=True)


def create_mask(shape, acc=2, dim='y'):
    mask = np.zeros(shape, dtype=DTYPE_real)

    if dim == 'x':
        mask[..., ::acc] = 1
    elif dim == 'y':
        mask[..., ::acc, :] = 1
    elif dim == 'z':
        mask[..., ::acc, :, :] = 1
    else:
        raise ValueError("Invalid dimension! Has to be x, y or z.")

    return mask


def undersample_kspace(par, ksp_data, args):
    acc = args.acceleration_factor
    dim = args.dim_us

    par["mask"] = np.zeros(ksp_data[0, 0, ...].shape, dtype=DTYPE_real)
    #mask = np.zeros(np.shape(ksp_data), dtype=DTYPE_real)
    mask = create_mask(np.shape(ksp_data), acc, dim)

    if dim == 'x':
        par["mask"][..., ::acc] = 1
    elif dim == 'y':
        par["mask"][..., ::acc, :] = 1
    elif dim == 'z':
        par["mask"][..., ::acc, :, :] = 1
    else:
        raise ValueError("Invalid dimension! Has to be x, y or z.")

    return ksp_data * mask


def sum_of_squares(x):
    return np.sqrt(np.abs(x[0])**2 + np.abs(x[1])**2) if x.shape[0] == 2 else np.abs(x)


def normalize_imgs(x):
    for i in range(x.shape[0]):
        img = x[i]
        i_min = np.min(img)
        i_max = np.max(img)
        x[i] = (img - i_min) / (i_max - i_min)
    return x


def calc_psnr(img, orig_img):
    rmse = calc_rmse(img, orig_img)
    psnr = 100
    if rmse != 0:
        psnr = 20 * np.log10(1. / rmse)
    return psnr


def calc_rmse(img, orig_img):
    return np.sqrt(calc_mse(img, orig_img))


def calc_mse(img, orig_img):
    return np.mean((img - orig_img)**2)


def calc_image_metrics(imgs, orig_imgs):
    mse = []
    psnr = []
    ssim = []

    for img, orig_img in zip(imgs, orig_imgs):
        mse.append(calc_mse(img, orig_img))
        psnr.append(calc_psnr(img, orig_img))
        ssim.append(calc_ssim(img, orig_img))

    print('MSE: ' + str(np.mean(mse)))
    print('PSNR: ' + str(np.mean(psnr)))
    print('SSIM: ' + str(np.mean(ssim)))

    return mse, psnr, ssim


def prepare_data(ksp, rescale=False, recon_type='2D'):
    shape = np.shape(ksp)
    z, y, x = shape[-3:]
    check = np.ones_like(ksp)
    check[..., 1::2] = -1
    check[..., ::2, :] *= -1
    if recon_type == '3D':
        check[..., ::2, :, :] *= -1

    result = ksp * check
    for nc in range(shape[0]):
        result[nc] = np.fft.ifftshift(result[nc])

    return result * np.sqrt(z * y * x) if rescale else result
