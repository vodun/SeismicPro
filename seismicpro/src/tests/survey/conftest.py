"""Survey fixtures with optionally collected stats"""

# pylint: disable=redefined-outer-name
import pytest

from seismicpro import Survey


@pytest.fixture(params=["TRACE_SEQUENCE_FILE", ("FieldRecord",), ["INLINE_3D", "CROSSLINE_3D"]])
def survey_no_stats(segy_path, request):
    """Return surveys with no stats collected."""
    return Survey(segy_path, header_index=request.param, header_cols="all")


@pytest.fixture
def survey_stats(survey_no_stats):
    """Return surveys with collected stats."""
    # copy is needed if both survey_no_stats and survey_stats are accessed by a single test or fixture
    return survey_no_stats.copy().collect_stats()


@pytest.fixture(params=[True, False])
def survey(survey_no_stats, request):
    """Return surveys with and without collected stats."""
    if request.param:
        # copy is needed since survey_no_stats will be updated inplace
        # and only surveys with collected stats will be returned
        return survey_no_stats.copy().collect_stats()
    return survey_no_stats
