"""Implements fixtures that """

import os

import pytest

from seismicpro.src import make_prestack_segy


@pytest.fixture(scope='package', autouse=True)
def segy_path(tmp_path_factory):
    """ Fixture that creates segy file """
    path = os.path.join(tmp_path_factory.getbasetemp(), 'test_prestack.sgy')
    make_prestack_segy(path, survey_size=(1000, 1000), origin=(0, 0), sources_step=(50, 300), recievers_step=(100, 25),
                       bin_size=(50, 50), activation_dist=(500, 500), samples=1500, sample_rate=2000, delay=0,
                       trace_gen=None)
    return path
