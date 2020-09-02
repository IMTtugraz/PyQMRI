#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:26:41 2019

@author: omaier
"""
import pytest
import os
import pyqmri
import shutil
import h5py
import numpy as np


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV():
    assert pyqmri.run(data=os.getcwd()+'/test/smalltest.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini'
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_imagespace_TGV():
    assert pyqmri.run(data=os.getcwd()+'/test/smalltest.h5',
                      model='VFA',
                      imagespace=True,
                      config=os.getcwd()+'/test/default.ini'
                      ) is None


@pytest.mark.integration_test
def test_General_model_kspace_TGV():
    assert pyqmri.run(data=os.getcwd()+'/test/smalltest.h5',
                      config=os.getcwd()+'/test/default.ini',
                      modelfile=os.getcwd()+'/test/models.ini'
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TV():
    assert pyqmri.run(data=os.getcwd()+'/test/smalltest.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      reg_type='TV'
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart():
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_smalltest.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_multislice(gen_multislice_data):
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_test.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      slices=4
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_multislice_streamed(gen_multislice_data):
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_test.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      slices=4,
                      streamed=1,
                      devices=0,
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_multislice_streamed(gen_multislice_data):
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_test.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      slices=4,
                      streamed=1,
                      devices=0,
                      reg_type='TV'
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_imageguess(
        gen_multislice_data_noimageguess):
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_smalltest.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_imageguess(
        gen_multislice_data_noimageguess):
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_smalltest.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      useCGguess=False,
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_coilguess(
        gen_multislice_data_nocoils):
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_test_coilguess.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      useCGguess=False,
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_coilguess_radial(
        gen_multislice_data_nocoils_radial):
    assert pyqmri.run(
        data=os.getcwd()+'/test/VFA_radial_test_coilguess.h5',
        model='VFA',
        config=os.getcwd()+'/test/default.ini',
        trafo=True,
        useCGguess=False,
        ) is None


@pytest.fixture(scope="function")
def gen_multislice_data():
    file = h5py.File(os.getcwd()+'/test/VFA_cart_smalltest.h5', 'r')

    Coils = file["Coils"][()]
    real_dat = file["real_dat"][()]
    imag_dat = file["imag_dat"][()]
    images = file["images"][()]
    fa_corr = file["fa_corr"][()]

    image_dimensions = file.attrs["image_dimensions"]
    fa = file.attrs["fa"][()]
    TR = file.attrs["TR"]

    file_out = h5py.File(os.getcwd()+'/test/VFA_cart_test.h5', 'w')

    slices = 4

    Coils = np.repeat(Coils, repeats=slices, axis=1)
    real_dat = np.repeat(real_dat, repeats=slices, axis=2)
    imag_dat = np.repeat(imag_dat, repeats=slices, axis=2)
    fa_corr = np.repeat(fa_corr, repeats=slices, axis=0)
    images = np.repeat(images, repeats=slices, axis=1)

    file_out["Coils"] = Coils
    file_out["real_dat"] = real_dat
    file_out["imag_dat"] = imag_dat
    file_out["fa_corr"] = fa_corr
    file_out["images"] = images

    image_dimensions[2] = slices

    file_out.attrs["TR"] = TR
    file_out.attrs["fa"] = fa
    file_out.attrs["flip_angle(s)"] = fa
    file_out.attrs["image_dimensions"] = image_dimensions
    file_out.close()
    file.close()

@pytest.fixture(scope="function")
def gen_multislice_data_noimageguess():
    file = h5py.File(os.getcwd()+'/test/VFA_cart_smalltest.h5', 'r')

    Coils = file["Coils"][()]
    real_dat = file["real_dat"][()]
    imag_dat = file["imag_dat"][()]
    fa_corr = file["fa_corr"][()]

    image_dimensions = file.attrs["image_dimensions"]
    fa = file.attrs["fa"][()]
    TR = file.attrs["TR"]

    file_out = h5py.File(os.getcwd()+'/test/VFA_cart_test_imageguess.h5', 'w')

    slices = 4

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
def gen_multislice_data_nocoils():
    file = h5py.File(os.getcwd()+'/test/VFA_cart_smalltest.h5', 'r')

    real_dat = file["real_dat"][()]
    imag_dat = file["imag_dat"][()]
    fa_corr = file["fa_corr"][()]

    image_dimensions = file.attrs["image_dimensions"]
    fa = file.attrs["fa"][()]
    TR = file.attrs["TR"]

    file_out = h5py.File(os.getcwd()+'/test/VFA_cart_test_coilguess.h5', 'w')

    slices = 4

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
def gen_multislice_data_nocoils_radial():
    file = h5py.File(os.getcwd()+'/test/smalltest.h5', 'r')

    real_dat = file["real_dat"][()]
    imag_dat = file["imag_dat"][()]
    real_traj = file["real_traj"][()]
    imag_traj = file["imag_traj"][()]
    fa_corr = file["fa_corr"][()]
    dcf_norm = file.attrs["data_normalized_with_dcf"]

    image_dimensions = file.attrs["image_dimensions"]
    fa = file.attrs["fa"][()]
    TR = file.attrs["TR"]

    file_out = h5py.File(os.getcwd()+'/test/VFA_radial_test_coilguess.h5', 'w')

    slices = 4

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
        shutil.rmtree(os.getcwd()+'/test/PyQMRI_out')
    except OSError as e:
        print ("Error: %s - %s." %(e.filename, e.strerror))
