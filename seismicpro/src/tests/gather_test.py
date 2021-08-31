"""Implementation of tests for survey"""
# pylint: disable=redefined-outer-name

import pytest
import numpy as np

from seismicpro import Survey, StackingVelocity


@pytest.fixture(scope='module')
def survey(segy_path):
    """Create gather"""
    survey = Survey(segy_path, header_index=['INLINE_3D', 'CROSSLINE_3D'], header_cols=['offset', 'FieldRecord'],
                    collect_stats=True)
    survey.headers['FirstBreak'] = np.random.randint(0, 1000, len(survey.headers))
    return survey

@pytest.fixture(scope='function')
def gather(survey):
    """gather"""
    return survey.get_gather((0, 0))

@pytest.mark.parametrize('tracewise,use_global', [[True, False], [False, False], [False, True]])
@pytest.mark.parametrize('q', [0.1, [0.1, 0.2], (0.1, 0.2), np.array([0.1, 0.2])])
def test_gather_get_quantile(gather, tracewise, use_global, q):
    """Test gahter's methods"""
    # # check that quantile has the same type as q
    gather.get_quantile(q=q, tracewise=tracewise, use_global=use_global)

@pytest.mark.parametrize('tracewise,use_global', [[True, False], [False, False], [False, True]])
def test_gather_scale_standard(gather, tracewise, use_global):
    """test_gather_scale_standard"""
    gather.scale_standard(tracewise=tracewise, use_global=use_global)

@pytest.mark.parametrize('tracewise,use_global', [[True, False], [False, False], [False, True]])
def test_gather_scale_minmax(gather, tracewise, use_global):
    """test_gather_scale_minmax"""
    gather.scale_minmax(tracewise=tracewise, use_global=use_global)

@pytest.mark.parametrize('tracewise,use_global', [[True, False], [False, False], [False, True]])
def test_gather_scale_maxabs(gather, tracewise, use_global):
    """test_gather_scale_minmax"""
    gather.scale_maxabs(tracewise=tracewise, use_global=use_global)

def test_gather_mask_to_pick_and_pick_to_mask(gather):
    """test_gather_mask_to_pick"""
    gather.pick_to_mask(first_breaks_col='FirstBreak', mask_attr='mask')
    gather.mask_to_pick(first_breaks_col='FirstBreak', mask_attr='mask')

def test_gather_get_coords(gather):
    """test_gather_get_coords"""
    gather.get_coords()

def test_gather_copy(gather):
    """test_gather_copy"""
    gather.copy()

def test_gather_sort(gather):
    """test_gather_sort"""
    gather.sort(by='offset')

def test_gather_validate(gather):
    """test_gather_validate"""
    gather.sort(by='offset')
    gather.validate(required_header_cols=['offset', 'FieldRecord'], required_sorting='offset')

def test_gather_muting(gather):
    """test_gather_muting"""
    offsets = [1000, 2000, 3000]
    times = [100, 300, 600]
    muter = gather.create_muter(mode='points', offsets=offsets, times=times)
    gather.mute(muter)

def test_gather_semblance(gather):
    """test_gather_semblance"""
    gather.sort(by='offset')
    velocities = np.linspace(1300, 5500, 140)
    gather.calculate_semblance(velocities=velocities)

def test_gather_res_semblance(gather):
    """test_gather_res_semblance"""
    gather.sort(by='offset')
    stacking_velocity = StackingVelocity.from_points(times=[0, 3000], velocities=[1600, 3500])
    gather.calculate_residual_semblance(stacking_velocity=stacking_velocity)

def test_gather_stacking_velocity(gather):
    """test_gather_stacking_velocity"""
    gather.sort(by='offset')
    stacking_velocity = StackingVelocity.from_points(times=[0, 3000], velocities=[1600, 3500])
    gather.apply_nmo(stacking_velocity=stacking_velocity)

def test_gather_get_central_cdp(segy_path):
    """test_gather_get_central_cdp"""
    survey = Survey(segy_path, header_index=['INLINE_3D', 'CROSSLINE_3D'], header_cols=['offset', 'FieldRecord'])
    survey = survey.generate_supergathers()
    gather = survey.get_gather((0, 0))
    gather.get_central_cdp()

def test_gather_stack(gather):
    """test_gather_stack"""
    gather.stack()
