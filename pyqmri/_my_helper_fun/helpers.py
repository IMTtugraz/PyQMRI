import numpy as np

DTYPE = np.complex64
DTYPE_real = np.float32


def gen_2ddata_from_imgs(imgs, cmaps):
    result = np.zeros(np.shape(cmaps)).astype(DTYPE)
    for z in range(np.shape(imgs)[1]):
        for c in range(np.shape(cmaps)[1]):
            for m in range(np.shape(cmaps)[0]):
                result[m, c, z, ...] = 1 * np.fft.ifftshift(np.fft.fft2(
                    np.fft.fftshift(imgs[m, z, ...] * cmaps[m, c, z, ...]), norm='ortho'))
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

def undersample_kspace(par, ksp_data, acc=2, dim='y'):
    par["mask"] = np.zeros(ksp_data[0, 0, ...].shape, dtype=DTYPE_real)
    #mask = np.zeros(np.shape(ksp_data), dtype=DTYPE_real)
    mask = create_mask(np.shape(ksp_data), acc, dim)

    if dim == 'x':
        par["mask"][..., ::acc] = 1
        # mask[..., ::acc] = 1
    elif dim == 'y':
        par["mask"][..., ::acc, :] = 1
        # mask[..., ::acc, :] = 1
    elif dim == 'z':
        par["mask"][..., ::acc, :, :] = 1
        # mask[..., ::acc, :, :] = 1
    else:
        raise ValueError("Invalid dimension! Has to be x, y or z.")

    return ksp_data * mask


def sum_of_squares(x):
    return np.sqrt(np.abs(x[0])**2 + np.abs(x[1])**2)


def normalize_imgs(x):
    for i in range(x.shape[0]):
        img = x[i]
        i_min = np.min(img)
        i_max = np.max(img)
        x[i] = (img - i_min) / (i_max - i_min)
    return x

def prepare_data(ksp, par):
    check = np.ones_like(ksp)
    check[..., 1::2] = -1
    check[..., ::2, :] *= -1
    if np.size(par["fft_dim"]) == 3:
        check[..., ::2, :, :] *= -1
    return ksp * check