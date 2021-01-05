import argparse

from pathlib import Path

from pyqmri._my_helper_fun.import_data import read_hdf
from pyqmri._my_helper_fun.display_data import img_montage
from pyqmri._my_helper_fun.helpers import *


class args:
    pass


def _main(args):

    for reg_type in ["NoReg", "TV", "TGV"]:
        if args.outdir == '':
            pathrootdir = Path.cwd() / 'server_output' / 'SoftSense_out' / reg_type
        else:
            raise NotImplementedError("For now just use correct directory structure...")
        dir_list = []
        for path in pathrootdir.iterdir():
            if path.is_dir():
                dir_list.append(path)

        if not dir_list:
            raise ValueError("Folder not found!")

        for dir in dir_list:
            path = pathrootdir / dir / 'data'
            f = path.glob('**/*')
            files = [x for x in f if x.is_file()]
            data = read_hdf(str(files[0]))
            img_montage(sqrt_sum_of_squares(data[0]), reg_type + '_' + str(files[0].name))


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description="Read and display hdf5 data from Soft sense reconstruction.")
    args.add_argument(
        '--outdir', default='', dest='outdir', type=str,
        help='Output directory.')
    args = args.parse_args()

    _main(args)
