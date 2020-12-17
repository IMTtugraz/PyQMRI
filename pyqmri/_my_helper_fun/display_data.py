import numpy as np
import matplotlib.pyplot as plt


def img_montage(imgs, title=''):
    if np.ndim(imgs) <= 3:
        imgs = np.expand_dims(imgs, axis=0)

    u = 1
    if np.ndim(imgs) == 4:
        u = np.shape(imgs)[0]

    for i in range(u):
        z, y, x = np.shape(imgs[i])

        xx = int(np.ceil(np.sqrt(z)))
        yy = xx
        montage = np.zeros((xx * x, yy * y))

        img_id = 0
        for m in range(xx):
            for n in range(yy):
                if img_id >= z:
                    break
                slice_n, slice_m = n * y, m * x
                montage[slice_m:slice_m + x, slice_n:slice_n + y] \
                    = np.flipud(imgs[i, img_id, :, :])
                img_id += 1

        # montage /= np.max(montage)
        # montage = np.clip(montage, 0, 1)

        fig = plt.figure()
        plt.imshow(montage, cmap='gray')
        plt.title(title)
        plt.show()