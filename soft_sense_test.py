# import matplotlib.pyplot as plt
# import numpy as np
import os

# from skimage.metrics import structural_similarity as calc_ssim

import pyqmri.softsense as softsense

from pyqmri._helper_fun._utils import gen_soft_sense_default_config


data_file_name = ''
cmaps_file_name = ''


# def img_montage(imgs, minval=None, maxval=None, title=''):
#     if np.ndim(imgs) <= 3:
#         imgs = np.expand_dims(imgs, axis=0)
#
#     u = 1
#     if np.ndim(imgs) == 4:
#         u = np.shape(imgs)[0]
#
#     for i in range(u):
#         z, y, x = np.shape(imgs[i])
#
#         xx = int(np.ceil(np.sqrt(z)))
#         yy = xx
#         montage = np.zeros((yy * y, xx * x))
#
#         img_id = 0
#         for m in range(xx):
#             for n in range(yy):
#                 if img_id >= z:
#                     break
#                 slice_n, slice_m = n * y, m * x
#                 montage[slice_n:slice_n + y, slice_m:slice_m + x] \
#                     = np.flipud(imgs[i, img_id, :, :])
#                 img_id += 1
#
#         vmin, vmax = 0, 1
#         if minval:
#             vmin = minval
#         if maxval:
#             vmax = maxval
#
#         plt.figure()
#         plt.imshow(montage, vmin=vmin, vmax=vmax, cmap='gray')
#         plt.title(title)
#         plt.show()
#
#
# def normalize_imgs(x):
#     if x.ndim == 2:
#         x = np.expand_dims(x, 0)
#     for i in range(x.shape[-3]):
#         img = x[i]
#         i_min = np.min(img)
#         i_max = np.max(img)
#         x[i] = (img - i_min) / (i_max - i_min) if np.abs(i_max - i_min) > 0 else 0
#     return np.squeeze(x)
#
#
# def calc_psnr(img, orig_img):
#     rmse = calc_rmse(img, orig_img)
#     psnr = 100
#     if rmse != 0:
#         psnr = 20 * np.log10(1. / rmse)
#     return psnr
#
#
# def calc_rmse(img, orig_img):
#     return np.sqrt(calc_mse(img, orig_img))
#
#
# def calc_mse(img, orig_img):
#     return np.mean((img - orig_img)**2)
#
#
# def calc_image_metrics(imgs, orig_imgs):
#     mse = []
#     psnr = []
#     ssim = []
#
#     for img, orig_img in zip(imgs, orig_imgs):
#         mse.append(calc_mse(img, orig_img))
#         psnr.append(calc_psnr(img, orig_img))
#         ssim.append(calc_ssim(img, orig_img))
#
#     print('-'*75)
#     print('MSE min: ' + str(np.min(mse)))
#     print('PSNR max: ' + str(np.max(psnr)))
#     print('SSIM max: ' + str(np.max(ssim)))
#     print('-' * 75)
#     return mse, psnr, ssim


def tv_entire_ds_test():
    _ = softsense.run(
        data=data_file_name,
        cmaps=cmaps_file_name,
        streamed=False,
        reg_type='TV',
        config='default_soft_sense.ini'
    )


def tgv_entire_ds_test():
    _ = softsense.run(
        data=data_file_name,
        cmaps=cmaps_file_name,
        streamed=False,
        reg_type='TGV',
        config='default_soft_sense.ini'
    )


def tv_entire_ds_test_streamed():
    _ = softsense.run(
        data=data_file_name,
        cmaps=cmaps_file_name,
        streamed=True,
        reg_type='TV',
        config='default_soft_sense.ini',
        par_slices=32
    )


def tgv_entire_ds_test_streamed():
    _ = softsense.run(
        data=data_file_name,
        cmaps=cmaps_file_name,
        streamed=True,
        reg_type='TGV',
        config='default_soft_sense.ini',
        par_slices=16
    )


def tv_chunk_ds_test():
    _ = softsense.run(
        data=data_file_name,
        cmaps=cmaps_file_name,
        streamed=False,
        reg_type='TV',
        config='default_soft_sense.ini',
        reco_slices=32,
        double_precision=False
    )


def tgv_chunk_ds_test():
    _ = softsense.run(
        data=data_file_name,
        cmaps=cmaps_file_name,
        streamed=False,
        reg_type='TGV',
        config='default_soft_sense.ini',
        reco_slices=32
    )


def tv_chunk_ds_test_streamed():
    _ = softsense.run(
        data=data_file_name,
        cmaps=cmaps_file_name,
        streamed=False,
        reg_type='TV',
        config='default_soft_sense.ini',
        reco_slices=64,
        par_slices=8
    )


def tgv_chunk_ds_test_streamed():
    _ = softsense.run(
        data=data_file_name,
        cmaps=cmaps_file_name,
        streamed=False,
        reg_type='TGV',
        config='default_soft_sense.ini',
        reco_slices=64,
        par_slices=8
    )


def main():
    if not os.path.exists(os.getcwd() + os.sep + 'default_soft_sense.ini'):
        gen_soft_sense_default_config()

    tv_chunk_ds_test()
    # tv_entire_ds_test()

    tgv_chunk_ds_test()
    # tgv_entire_ds_test()

    # tv_entire_ds_test_streamed()
    # tgv_entire_ds_test_streamed()

    # tv_chunk_ds_test_streamed()
    # tgv_chunk_ds_test_streamed()


if __name__ == '__main__':
    main()
