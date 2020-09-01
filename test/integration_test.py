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
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_test.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_multislice():
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_test.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      slices=4
                      ) is None


@pytest.mark.integration_test
def test_VFA_model_kspace_TGV_cart_multislice_streamed():
    assert pyqmri.run(data=os.getcwd()+'/test/VFA_cart_test.h5',
                      model='VFA',
                      config=os.getcwd()+'/test/default.ini',
                      trafo=False,
                      slices=4,
                      streamed=1,
                      devices=0,
                      ) is None


@pytest.fixture(autouse=True, scope="session")
def clean_up():
    yield
    try:
        shutil.rmtree(os.getcwd()+'/test/PyQMRI_out')
    except OSError as e:
        print ("Error: %s - %s." %(e.filename, e.strerror))
