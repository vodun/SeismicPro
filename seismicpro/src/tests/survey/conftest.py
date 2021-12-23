"""Survey fixtures with optionally collected stats"""

import pytest

from seismicpro import Survey


@pytest.fixture(params=["TRACE_SEQUENCE_FILE", ("FieldRecord",), ["INLINE_3D", "CROSSLINE_3D"]])
def survey_no_stats(segy_path, request):
    return Survey(segy_path, header_index=request.param, header_cols="all")


@pytest.fixture
def survey_stats(survey_no_stats):
    return survey_no_stats.copy().collect_stats()


@pytest.fixture(params=[True, False])
def survey(survey_no_stats, request):
    if request.param:
        return survey_no_stats.copy().collect_stats()
    return survey_no_stats
