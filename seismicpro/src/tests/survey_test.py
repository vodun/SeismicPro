"""Implementation of tests for survey"""

import pytest
import numpy as np

from seismicpro import Survey

# TODO:
# 1. Add first break file for test
# 2. Add raises for cases that not covers with `if`
@pytest.mark.parametrize('header_cols', [None, 'offset', 'all'])
@pytest.mark.parametrize('name', [None, 'raw'])
@pytest.mark.parametrize('limits', [None, (0, 1000)])
@pytest.mark.parametrize('collect_stats', [True, False])
def test_survey_methods(segy_path, header_cols, name, limits, collect_stats):
    """Test survey's methods"""
    survey = Survey(segy_path, header_index='FieldRecord', header_cols=header_cols, name=name, limits=limits,
                    collect_stats=collect_stats)
    survey.info()
    if survey.has_stats:
        _ = survey.get_quantile(0.01)
    _ = survey.get_gather(index=survey.headers.index[0])
    _ = survey.copy()
    if 'offset' in survey.headers.columns:
        _ = survey.filter(lambda offset: offset < 1500, cols="offset", inplace=False)
        _ = survey.apply(lambda offset: np.abs(offset), cols="offset", inplace=False)
    _ = survey.reindex('TRACE_SEQUENCE_FILE', inplace=False)
    sur = survey.copy()
    _ = sur.set_limits(1000)
    _ = sur.set_limits((1, 1000))
    _ = sur.set_limits((1, 1000, 2))
    _ = sur.set_limits(slice(0, 1000, 2))
    if 'INLINE_3D' in survey.headers.columns and 'CROSSLINE_3D' in survey.headers.columns:
        _ = sur.generate_supergathers()
