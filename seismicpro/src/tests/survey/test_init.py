"""Test Survey class instantiation"""

import pytest
import segyio
import numpy as np

from seismicpro import Survey

from . import assert_survey_loaded, assert_surveys_equal, assert_survey_limits
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

    def test_nolimits_nostats(self, segy_path, header_index, expected_index, header_cols, expected_cols,
                              name, expected_name):
        """Test survey loading when limits are not passed and stats are not calculated."""
        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name)

        expected_headers = expected_index | expected_cols | {"TRACE_SEQUENCE_FILE"}
        assert_survey_loaded(survey, segy_path, expected_name, expected_index, expected_headers)

        # Assert that whole traces are loaded
        limits = slice(0, survey.n_file_samples, 1)
        assert_survey_limits(survey, limits)

        # Assert that stats are not calculated
        assert survey.has_stats is False

    @pytest.mark.parametrize(["limits", "slice_limits"], LIMITS)
    def test_limits_nostats(self, segy_path, header_index, expected_index, header_cols, expected_cols,
                            name, expected_name, limits, slice_limits):
        """Test survey loading with limits set when stats are not calculated."""
        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name, limits=limits)

        expected_headers = expected_index | expected_cols | {"TRACE_SEQUENCE_FILE"}
        assert_survey_loaded(survey, segy_path, expected_name, expected_index, expected_headers)

        # Assert that correct limits were set
        assert_survey_limits(survey, slice_limits)

        # Assert that stats are not calculated
        assert survey.has_stats is False

        # Check that passing limits to init is identical to running set_limits method
        other = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name)
        other.set_limits(limits)
        assert_surveys_equal(survey, other)

    @pytest.mark.parametrize("n_quantile_traces", [10])
    @pytest.mark.parametrize("quantile_precision", [1])
    @pytest.mark.parametrize("stats_limits", [None, 100])
    def test_nolimits_stats(self, segy_path, header_index, expected_index,  # pylint: disable=too-many-arguments
                            header_cols, expected_cols, name, expected_name, n_quantile_traces, quantile_precision,
                            stats_limits, monkeypatch):
        """Test survey loading when stats are calculated, but limits are not set."""
        # Always use the same traces for quantile estimation
        monkeypatch.setattr(np.random, "permutation", lambda n: np.arange(n))

        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name,
                        collect_stats=True, n_quantile_traces=n_quantile_traces, quantile_precision=quantile_precision,
                        stats_limits=stats_limits, bar=False)

        # Assert that a survey was loaded correctly and an extra DeadTrace header was created
        expected_headers = expected_index | expected_cols | {"TRACE_SEQUENCE_FILE", "DeadTrace"}
        assert_survey_loaded(survey, segy_path, expected_name, expected_index, expected_headers)

        # Assert that whole traces are loaded
        limits = slice(0, survey.n_file_samples, 1)
        assert_survey_limits(survey, limits)

        # Assert that stats are calculated
        assert survey.has_stats is True

        # Check that passing collect_stats to init is identical to running collect_stats method
        other = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name)
        other.collect_stats(n_quantile_traces=n_quantile_traces, quantile_precision=quantile_precision,
                            stats_limits=stats_limits, bar=False)
        assert_surveys_equal(survey, other)

    @pytest.mark.parametrize("n_quantile_traces", [10])
    @pytest.mark.parametrize("quantile_precision", [1])
    @pytest.mark.parametrize("stats_limits", [None, 100])
    @pytest.mark.parametrize(["limits", "slice_limits"], LIMITS)
    def test_limits_stats(self, segy_path, header_index, expected_index,  # pylint: disable=too-many-arguments
                          header_cols, expected_cols, name, expected_name, limits, slice_limits, n_quantile_traces,
                          quantile_precision, stats_limits, monkeypatch):
        """Test survey loading when limits are set and stats are calculated."""
        # Always use the same traces for quantile estimation
        monkeypatch.setattr(np.random, "permutation", lambda n: np.arange(n))

        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name, limits=limits,
                        collect_stats=True, n_quantile_traces=n_quantile_traces, quantile_precision=quantile_precision,
                        stats_limits=stats_limits, bar=False)

        # Assert that a survey was loaded correctly and an extra DeadTrace header was created
        expected_headers = expected_index | expected_cols | {"TRACE_SEQUENCE_FILE", "DeadTrace"}
        assert_survey_loaded(survey, segy_path, expected_name, expected_index, expected_headers)

        # Assert that correct limits were set
        assert_survey_limits(survey, slice_limits)

        # Assert that stats are calculated
        assert survey.has_stats is True

        # Check that passing limits and collect_stats to init is identical to first running set_limits with the
        # following run of collect_stats
        other = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name)
        other.set_limits(limits)
        other.collect_stats(n_quantile_traces=n_quantile_traces, quantile_precision=quantile_precision,
                            stats_limits=stats_limits, bar=False)
        assert_surveys_equal(survey, other)
