import argparse

from pathlib import Path

from pyqmri._my_helper_fun.import_data import read_hdf
from pyqmri._my_helper_fun.display_data import img_montage

class args:
    pass


def main():
    filename = args.type + '_recon_' + args.reg_type + '_lambda_' + args.lamda + '.hdf5'
    pathfilename = Path.cwd() / 'output' / 'data' / filename
    pathfile = Path(pathfilename)

    data = read_hdf(pathfile)[0]
    img_montage(data)


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description="Read and display hdf5 data from Soft sense reconstruction.")
    args.add_argument(
      '--recon_type', default='2D', dest='type',
      help='Choose reconstruction type, 2D or 3D')
    args.add_argument(
      '--reg_type', default='', dest='reg_type',
      help="Choose regularization type (default: without regularization) "
           "options are: 'TGV', 'TV', ''")
    args.add_argument(
        '--lambda', default=1, dest='lamda',
        help="Regularization parameter (default: 1)", type=float
    )

    args = args.parse_args()

    args.type = '2D'
    args.reg_type = ''
    args.lamda = '1e-01'
    main()
