""" Tests for segy generator function """
# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

import os
import shutil

import numpy as np
import pytest
from ..utils import make_prestack_segy
from .. import Survey

@pytest.fixture(scope='module',
                params=[dict(sources_size=500, activation_dist=500, bin_size=50, samples=1500,
                             dist_source_lines=300, dist_sources=50, dist_reciever_lines=100, dist_recievers=25),
                        dict(sources_size=1200, activation_dist=300, bin_size=30, samples=100,
                             dist_source_lines=200, dist_sources=50, dist_reciever_lines=100, dist_recievers=25),
                        dict(sources_size=100, activation_dist=100, bin_size=100, samples=1000,
                             dist_source_lines=300, dist_sources=50, dist_reciever_lines=200, dist_recievers=50)])
def files_setup(request):
    """ Fixture that creates segy file """
    folder = 'test_tmp'

    os.mkdir(folder)
    path = os.path.join(folder, 'test_prestack.sgy')
    make_prestack_segy(path, **request.param)

    def fin():
        shutil.rmtree(folder)

    request.addfinalizer(fin)
    return path

@pytest.mark.parametrize('header_index',
                         ('FieldRecord', ['INLINE_3D', 'CROSSLINE_3D'], ['GroupX', 'GroupY'], ['SourceX', 'SourceY']))
def test_generated_segy_loading(files_setup, header_index):
    s = Survey(files_setup, header_index=header_index, header_cols=['FieldRecord', 'TraceNumber', 'SourceX', 'SourceY',
                                                                    'GroupX', 'GroupY', 'offset', 'CDP_X',
                                                                    'CDP_Y', 'INLINE_3D', 'CROSSLINE_3D'])
    assert s.sample_gather()
