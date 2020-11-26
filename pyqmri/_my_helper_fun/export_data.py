import h5py
import numpy as np
from PIL import Image
from pathlib import Path

DTYPE = np.complex64
DTYPE_real = np.float32


def save_data(data, par, irgn_par, myargs, filename='', ds_name='dataset', store_attrs=True):
    par_keys = ['dimX', 'dimY', 'NSlice', 'NC', 'NScan', 'overlap', 'par_slices']
    irgn_par_keys = ['sigma', 'lambd', 'alpha0', 'alpha1', 'delta', 'gamma', 'accelerated']

    if filename == '':
        filename = myargs.type + '_recon_' + myargs.reg_type + '_lambda_' + '{:.0e}'.format(myargs.lamda)

    path = Path.cwd() / 'output' / 'data'
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    file_dir = path / (filename + '.hdf5')

    cnt = 0
    while file_dir.exists():
        file_dir = path / (filename + '_' + str(cnt) + '.hdf5')
        cnt += 1

    with h5py.File(file_dir, 'w') as f:
        dset = f.create_dataset(ds_name, data=data)
        if store_attrs:
            for key in par_keys:
                dset.attrs[key] = par[key]
            for key in irgn_par_keys:
                dset.attrs[key] = irgn_par[key]
            dset.attrs['acc_factor'] = myargs.acceleration_factor
            dset.attrs['linesearch'] = myargs.linesearch
            dset.attrs['streamed'] = myargs.streamed


def save_imgs(imgs, myargs, filename=''):
    if filename == '':
        filename = myargs.type + '_recon_' + myargs.reg_type + '_lambda_' + '{:.0e}'.format(myargs.lamda)

    path = Path.cwd() / 'output' / 'imgs'
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        img = np.flipud(np.abs(img))
        img_rescaled = (255.0 / img.max() * (img - img.min())).astype(np.uint8)

        im = Image.fromarray(img_rescaled)
        file_dir = path / (filename + '_' + str(i) + '.png')
        im.save(file_dir)
