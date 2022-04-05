"""Test Survey class instantiation"""

import pytest
import segyio

from seismicpro import Survey

from . import assert_survey_loaded, assert_surveys_equal, assert_survey_limits_set
from ..conftest import FILE_NAME, N_SAMPLES


HEADER_INDEX = [
    # Single header index passed as a string, list or tuple:
    ["TRACE_SEQUENCE_FILE", {"TRACE_SEQUENCE_FILE"}],
    [("FieldRecord",), {"FieldRecord"}],
    [["CDP"], {"CDP"}],

    # Multiple header indices passed as a list or tuple:
    [["FieldRecord", "TraceNumber"], {"FieldRecord", "TraceNumber"}],
    [("INLINE_3D", "CROSSLINE_3D"), {"INLINE_3D", "CROSSLINE_3D"}],
]


HEADER_COLS = [
    # Don't load extra headers
    [None, set()],

    # Load all SEG-Y headers
    ["all", set(segyio.tracefield.keys.keys())],

    # Load a single extra header passed as a string, list or tuple:
    ["offset", {"offset"}],
    [["offset"], {"offset"}],
    [("offset",), {"offset"}],

    # Load several extra headers passed as a list or a tuple:
    [["offset", "SourceDepth"], {"offset", "SourceDepth"}],
    [("offset", "SourceDepth"), {"offset", "SourceDepth"}],

    # Load several extra headers with possible intersection with index
    [["offset", "INLINE_3D", "CROSSLINE_3D"], {"offset", "INLINE_3D", "CROSSLINE_3D"}],
]


NAME = [
    # Use file name if survey name is not passed
    [None, FILE_NAME],

    # Use passed name otherwise
    ["raw", "raw"],
]


LIMITS = [
    (None, slice(0, N_SAMPLES, 1)),
    (10, slice(0, 10, 1)),
    (slice(100, -100), slice(100, N_SAMPLES - 100, 1)),
]


@pytest.mark.parametrize("header_index, expected_index", HEADER_INDEX)
@pytest.mark.parametrize("header_cols, expected_cols", HEADER_COLS)
@pytest.mark.parametrize("name, expected_name", NAME)
class TestInit:
    """Test `Survey` instantiation."""

    @pytest.mark.parametrize("chunk_size, n_workers, bar", [
        [1, 1, True],  # Single worker, tracewise loading with bar
        [10, None, True],  # os.cpu_count() workers, small chunk size with bar
        [10000000, None, False],  # Chunk size larger than the number of traces
    ])
    def test_nolimits(self, segy_path, header_index, expected_index, header_cols, expected_cols,
                      name, expected_name, chunk_size, n_workers, bar):
        """Test survey loading when limits are not passed."""
        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name,
                        chunk_size=chunk_size, n_workers=n_workers, bar=bar)

        expected_headers = expected_index | expected_cols | {"TRACE_SEQUENCE_FILE"}
        assert_survey_loaded(survey, segy_path, expected_name, expected_index, expected_headers)

        # Assert that whole traces are loaded
        limits = slice(0, survey.n_file_samples, 1)
        assert_survey_limits_set(survey, limits)

        # Assert that stats are not calculated
        assert survey.has_stats is False
        assert survey.dead_traces_marked is False

    @pytest.mark.parametrize(["limits", "slice_limits"], LIMITS)
    def test_limits(self, segy_path, header_index, expected_index, header_cols, expected_cols,
                    name, expected_name, limits, slice_limits):
        """Test survey loading with limits set."""
        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name, limits=limits,
                        bar=False)

        expected_headers = expected_index | expected_cols | {"TRACE_SEQUENCE_FILE"}
        assert_survey_loaded(survey, segy_path, expected_name, expected_index, expected_headers)

        # Assert that correct limits were set
        assert_survey_limits_set(survey, slice_limits)

        # Assert that stats are not calculated
        assert survey.has_stats is False
        assert survey.dead_traces_marked is False

        # Check that passing limits to init is identical to running set_limits method
        other = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name)
        other.set_limits(limits)
        assert_surveys_equal(survey, other)
