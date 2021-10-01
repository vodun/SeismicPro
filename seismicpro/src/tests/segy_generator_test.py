""" Tests for segy generator function """
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name

import os
import shutil

import pytest
from ..utils import make_prestack_segy
from .. import Survey

@pytest.fixture(scope='module',
                params=[dict(origin=(0,0), n_samples=1500),
                        dict(origin=(100,100), n_samples=100),
                        dict(origin=(-100,-100), n_samples=1000)])
def create_segy(request):
    """ Fixture that creates segy file """
    folder = 'test_tmp'

    os.mkdir(folder)
    path = os.path.join(folder, 'test_prestack.sgy')
    make_prestack_segy(path, **request.param)

    def fin():
        shutil.rmtree(folder)

    request.addfinalizer(fin)
    return path, request.param['origin'], request.param['n_samples']

@pytest.mark.parametrize('header_index',
                         ('FieldRecord', ['INLINE_3D', 'CROSSLINE_3D'], ['GroupX', 'GroupY'], ['SourceX', 'SourceY']))
def test_generated_segy_loading(create_segy, header_index):
    segy_path, origin, n_samples = create_segy
    s = Survey(segy_path, header_index=header_index, header_cols=['FieldRecord', 'TraceNumber', 'SourceX', 'SourceY',
                                                                  'GroupX', 'GroupY', 'offset', 'CDP_X', 'CDP_Y',
                                                                  'INLINE_3D', 'CROSSLINE_3D'])
    assert s.sample_gather()
    assert s.samples_length == n_samples
    assert s.headers.reset_index().GroupX.min() == origin[0]
