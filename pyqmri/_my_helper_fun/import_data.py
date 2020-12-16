import sys
import numpy as np
import h5py
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.io import loadmat
from pathlib import Path


def read_hdf(pathfile):
    data_sets = []
    with h5py.File(pathfile, 'r') as f:
        for key in f.keys():
            data_sets.append(f[key][()])
    f.close()
    return data_sets


def _read_unknown(pathfile):
    print('Can not read ' + pathfile.name + '. Unknown file extension! Returning empty array.')
    return np.empty([])


def _read_mat(pathfile):
    data_sets = []
    try:
        f = loadmat(pathfile)
        key_list = list(f.keys())[3:]
        for key in key_list:
            data_sets.append(f[key][()])
        return data_sets
    except NotImplementedError:
        with h5py.File(pathfile, 'r') as f:
            for key in f.keys():
                data_sets.append(f[key][()])
        f.close()
        return data_sets
    except:
        print('Can not read ' + pathfile.name + '. Exiting...')
        sys.exit()


def import_data(pathfile, dialog):
    """Imports datasets with various extensions.

        Parameters:
        pathfile -- Path file object

        Returns:
        datasets(list) -- List of datasets read from file
    """

    if pathfile == '':
        root = Tk()
        root.withdraw()
        root.update()
        root.geometry('200x100')
        root.title('Select ' + dialog + ' data')

        filename = askopenfilename()
        root.destroy()
        if filename == ():
            print('No file selected. Exiting...')
            sys.exit()
        pathfile = Path(filename)

    file_ext = pathfile.suffix
    data = {'.mat': _read_mat(pathfile)#,
            #'hdf5': _read_hdf(pathfile)
            }.get(file_ext)

    if data is not None:
        return data
    return _read_unknown(pathfile)