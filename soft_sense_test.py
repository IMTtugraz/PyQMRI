import os
import pyqmri.softsense as softsense

from pyqmri._helper_fun._utils import gen_soft_sense_default_config


def main():
    if not os.path.exists(os.getcwd() + os.sep + 'default_soft_sense.ini'):
        gen_soft_sense_default_config()

    data_file_name = '/home/christoph/Studium/Masterthesis/PyQMRI/Granat/Results/C2x2s0/kspace_sc_cc.mat'
    cmaps_file_name = '/home/christoph/Studium/Masterthesis/PyQMRI/Granat/Results/C2x2s0/sensitivities_ecalib.mat'
    mask_file_name = '/home/christoph/Studium/Masterthesis/PyQMRI/Granat/Results/C2x2s0/mask_C2x2s0.mat'
    softsense.run(
        data=data_file_name,
        cmaps=cmaps_file_name,
        mask=mask_file_name,
        streamed=False,
        reg_type='TV',
        config='default_soft_sense.ini')


if __name__ == '__main__':
    main()
