#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:26:41 2019

@author: omaier
"""
import pytest
import os
from os.path import join as pjoin
import pyqmri
import shutil
import h5py
import numpy as np

data_dir = os.path.realpath(pjoin(os.path.dirname(__file__), '..'))

@pytest.mark.integration_test
def test_VFA_model_kspace_TGV():
    assert pyqmri.run(data=pjoin(data_dir, 'smalltest.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_imagespace_TGV():
    assert pyqmri.run(data=pjoin(data_dir, 'smalltest.h5'),
                      model='VFA',
                      imagespace=True,
                      config=pjoin(data_dir, 'default.ini'),
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_General_model_kspace_TGV():
    assert pyqmri.run(data=pjoin(data_dir, 'smalltest.h5'),
                      config=pjoin(data_dir, 'default.ini'),
                      modelfile=pjoin(data_dir, 'models.ini'),
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TV():
    assert pyqmri.run(data=pjoin(data_dir, 'smalltest.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      reg_type='TV',
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart():
    assert pyqmri.run(data=pjoin(data_dir, 'VFA_cart_smalltest.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      trafo=False,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_imageguess_CG(
        gen_noimageguess):
    assert pyqmri.run(data=pjoin(data_dir, 'VFA_cart_test_imageguess.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      trafo=False,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_imageguess(
        gen_noimageguess):
    assert pyqmri.run(data=pjoin(data_dir, 'VFA_cart_test_imageguess.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      trafo=False,
                      useCGguess=False,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_coilguess(
        gen_data_nocoils):
    assert pyqmri.run(data=pjoin(data_dir, 'VFA_cart_test_coilguess.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      trafo=False,
                      useCGguess=False,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_coilguess_radial(
        gen_data_nocoils_radial):
    assert pyqmri.run(
        data=pjoin(data_dir, 'VFA_radial_test_coilguess.h5'),
        model='VFA',
        config=pjoin(data_dir, 'default.ini'),
        trafo=True,
        useCGguess=False,
        use_GPU=False
        ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_double():
    assert pyqmri.run(data=pjoin(data_dir, 'smalltest.h5'),
                      model='VFA',
                      double_precision=True,
                      config=pjoin(data_dir, 'default.ini'),
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_imagespace_TGV_double():
    assert pyqmri.run(data=pjoin(data_dir, 'smalltest.h5'),
                      model='VFA',
                      imagespace=True,
                      double_precision=True,
                      config=pjoin(data_dir, 'default.ini'),
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_General_model_kspace_TGV_double():
    assert pyqmri.run(data=pjoin(data_dir, 'smalltest.h5'),
                      config=pjoin(data_dir, 'default.ini'),
                      modelfile=pjoin(data_dir, 'models.ini'),
                      double_precision=True,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TV_double():
    assert pyqmri.run(data=pjoin(data_dir, 'smalltest.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      reg_type='TV',
                      double_precision=True,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_double():
    assert pyqmri.run(data=pjoin(data_dir, 'VFA_cart_smalltest.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      trafo=False,
                      double_precision=True,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_imageguess_CG_double(
        gen_noimageguess):
    assert pyqmri.run(data=pjoin(data_dir, 'VFA_cart_test_imageguess.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      trafo=False,
                      double_precision=True,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_imageguess_double(
        gen_noimageguess):
    assert pyqmri.run(data=pjoin(data_dir, 'VFA_cart_test_imageguess.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      trafo=False,
                      useCGguess=False,
                      double_precision=True,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_coilguess_double(
        gen_data_nocoils):
    assert pyqmri.run(data=pjoin(data_dir, 'VFA_cart_test_coilguess.h5'),
                      model='VFA',
                      config=pjoin(data_dir, 'default.ini'),
                      trafo=False,
                      useCGguess=False,
                      double_precision=True,
                      use_GPU=False
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_coilguess_radial_double(
        gen_data_nocoils_radial):
    assert pyqmri.run(
        data=pjoin(data_dir, 'VFA_radial_test_coilguess.h5'),
        model='VFA',
        config=pjoin(data_dir, 'default.ini'),
        trafo=True,
        useCGguess=False,
        double_precision=True,
        use_GPU=False
        ) is None

@pytest.fixture(scope="function")
def gen_noimageguess():
    file = h5py.File(pjoin(data_dir, 'VFA_cart_smalltest.h5'), 'r')

    Coils = file["Coils"][()]
    real_dat = file["real_dat"][()]
    imag_dat = file["imag_dat"][()]
    fa_corr = file["fa_corr"][()]

    image_dimensions = file.attrs["image_dimensions"]
    fa = file.attrs["fa"][()]
    TR = file.attrs["TR"]

    file_out = h5py.File(pjoin(data_dir, 'VFA_cart_test_imageguess.h5'), 'w')

    slices = 1

    Coils = np.repeat(Coils, repeats=slices, axis=1)
    real_dat = np.repeat(real_dat, repeats=slices, axis=2)
    imag_dat = np.repeat(imag_dat, repeats=slices, axis=2)
    fa_corr = np.repeat(fa_corr, repeats=slices, axis=0)

    file_out["Coils"] = Coils
    file_out["real_dat"] = real_dat
    file_out["imag_dat"] = imag_dat
    file_out["fa_corr"] = fa_corr

    image_dimensions[2] = slices

    file_out.attrs["TR"] = TR
    file_out.attrs["fa"] = fa
    file_out.attrs["flip_angle(s)"] = fa
    file_out.attrs["image_dimensions"] = image_dimensions
    file_out.close()
    file.close()


@pytest.fixture(scope="function")
def gen_data_nocoils():
    file = h5py.File(pjoin(data_dir, 'VFA_cart_smalltest.h5'), 'r')

    real_dat = file["real_dat"][()]
    imag_dat = file["imag_dat"][()]
    fa_corr = file["fa_corr"][()]

    image_dimensions = file.attrs["image_dimensions"]
    fa = file.attrs["fa"][()]
    TR = file.attrs["TR"]

    file_out = h5py.File(pjoin(data_dir, 'VFA_cart_test_coilguess.h5'), 'w')

    slices = 1

    real_dat = np.repeat(real_dat, repeats=slices, axis=2)
    imag_dat = np.repeat(imag_dat, repeats=slices, axis=2)
    fa_corr = np.repeat(fa_corr, repeats=slices, axis=0)

    file_out["real_dat"] = real_dat
    file_out["imag_dat"] = imag_dat
    file_out["fa_corr"] = fa_corr

    image_dimensions[2] = slices

    file_out.attrs["TR"] = TR
    file_out.attrs["fa"] = fa
    file_out.attrs["flip_angle(s)"] = fa
    file_out.attrs["image_dimensions"] = image_dimensions
    file_out.close()
    file.close()


@pytest.fixture(scope="function")
def gen_data_nocoils_radial():
    file = h5py.File(pjoin(data_dir, 'smalltest.h5'), 'r')

    real_dat = file["real_dat"][()]
    imag_dat = file["imag_dat"][()]
    real_traj = file["real_traj"][()]
    imag_traj = file["imag_traj"][()]

    fa_corr = file["fa_corr"][()]
    dcf_norm = file.attrs["data_normalized_with_dcf"]

    image_dimensions = file.attrs["image_dimensions"]
    fa = file.attrs["fa"][()]
    TR = file.attrs["TR"]

    file_out = h5py.File(pjoin(data_dir, 'VFA_radial_test_coilguess.h5'), 'w')

    slices = 1

    real_dat = np.repeat(real_dat, repeats=slices, axis=2)
    imag_dat = np.repeat(imag_dat, repeats=slices, axis=2)
    fa_corr = np.repeat(fa_corr, repeats=slices, axis=0)

    file_out["real_dat"] = real_dat
    file_out["imag_dat"] = imag_dat
    file_out["real_traj"] = real_traj
    file_out["imag_traj"] = imag_traj
    file_out["fa_corr"] = fa_corr

    image_dimensions[2] = slices

    file_out.attrs["TR"] = TR
    file_out.attrs["fa"] = fa
    file_out.attrs["flip_angle(s)"] = fa
    file_out.attrs["image_dimensions"] = image_dimensions
    file_out.attrs["data_normalized_with_dcf"] = dcf_norm
    file_out.close()
    file.close()


@pytest.fixture(autouse=True, scope="session")
def clean_up():
    yield
    try:
        if os.path.exists(pjoin(data_dir, 'PyQMRI_out')):
            shutil.rmtree(pjoin(data_dir, 'PyQMRI_out'))
        if os.path.isfile(pjoin(data_dir, 'VFA_cart_test_imageguess.h5')):
            os.remove(pjoin(data_dir, 'VFA_cart_test_imageguess.h5'))
        if os.path.isfile(pjoin(data_dir, 'VFA_cart_test_coilguess.h5')):
            os.remove(pjoin(data_dir, 'VFA_cart_test_coilguess.h5'))
        if os.path.isfile(pjoin(data_dir, 'VFA_radial_test_coilguess.h5')):
            os.remove(pjoin(data_dir, 'VFA_radial_test_coilguess.h5'))
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
