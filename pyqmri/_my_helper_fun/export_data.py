import h5py
import numpy as np
from PIL import Image
from pathlib import Path

DTYPE = np.complex64
DTYPE_real = np.float32


# def save_data(data, par, ds_name='dataset', filename='def'):
#     path = Path.cwd() / 'output' / 'data'
#     if not path.exists():
#         path.mkdir(parents=True, exist_ok=True)
#     file_dir = path / (filename + '.hdf5')
#     with h5py.File(file_dir, 'w') as f:
#         dset = f.create_dataset(ds_name, data=data)
#         #for key in par.keys():
#         #    dset.attrs[key] = par[key]


def save_imgs(imgs, filename='def'):
    path = Path.cwd() / 'output' / 'imgs'
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        img = np.flipud(np.abs(img))
        img_rescaled = (255.0 / img.max() * (img - img.min())).astype(np.uint8)

        im = Image.fromarray(img_rescaled)
        file_dir = path / (filename + '_' + str(i) + '.png')
        im.save(file_dir)
