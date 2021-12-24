"""Test Survey class instantiation"""

import pytest
import numpy as np

from seismicpro import Survey

from . import assert_surveys_equal, assert_survey_limits, assert_survey_loaded


HEADER_INDEX = [
    # Single header as a string, list and tuple:
    "TRACE_SEQUENCE_FILE",
    ("FieldRecord",),
    ["CDP"],

    # Multiple headers as a list or tuple:
    ["FieldRecord", "TraceNumber"],
    ("INLINE_3D", "CROSSLINE_3D"),
]


HEADER_COLS = [
    None,  # Don't load extra headers
    "all",  # Load all SEG-Y headers
    "offset",  # Load a single header
    ["offset"],  # Load a single header (pass as a list)
    ["offset", "SourceDepth"],  # Load several headers (pass as a list)
    ("offset", "SourceDepth"),  # Load several headers (pass as a tuple)
    ["offset", "INLINE_3D", "CROSSLINE_3D"],  # Load several headers with possible intersetion with index
]


LIMITS = [
    (None, slice(0, 1000, 1)),
    (10, slice(0, 10, 1)),
    (slice(100, -100), slice(100, 900, 1)),
]


@pytest.mark.parametrize("header_index", HEADER_INDEX)
@pytest.mark.parametrize("header_cols", HEADER_COLS)
@pytest.mark.parametrize("name", [None, "raw"])
class TestInit:
    def test_nolimits_nostats(self, segy_path, header_index, header_cols, name):
        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name)
        assert_survey_loaded(survey, segy_path, header_index, header_cols, name)

        # Assert that whole traces are loaded
        limits = slice(0, survey.n_file_samples, 1)
        assert_survey_limits(survey, limits)

        # Assert that stats are not calculated
        assert survey.has_stats is False

    @pytest.mark.parametrize(["limits", "slice_limits"], LIMITS)
    def test_limits_nostats(self, segy_path, header_index, header_cols, name, limits, slice_limits):
        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name, limits=limits)
        assert_survey_loaded(survey, segy_path, header_index, header_cols, name)

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
    def test_nolimits_stats(self, segy_path, header_index, header_cols, name, n_quantile_traces, quantile_precision,
                            stats_limits, monkeypatch):
        # Always use the same traces for quantile estimation
        monkeypatch.setattr(np.random, "permutation", lambda n: np.arange(n))

        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name,
                        collect_stats=True, n_quantile_traces=n_quantile_traces, quantile_precision=quantile_precision,
                        stats_limits=stats_limits, bar=False)

        # Assert that a survey was loaded correctly and an extra DeadTrace header was created
        assert_survey_loaded(survey, segy_path, header_index, header_cols, name, extra_headers="DeadTrace")

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
    def test_limits_stats(self, segy_path, header_index, header_cols, name,  # pylint: disable=too-many-arguments
                          limits, slice_limits, n_quantile_traces, quantile_precision, stats_limits, monkeypatch):
        # Always use the same traces for quantile estimation
        monkeypatch.setattr(np.random, "permutation", lambda n: np.arange(n))

        survey = Survey(segy_path, header_index=header_index, header_cols=header_cols, name=name, limits=limits,
                        collect_stats=True, n_quantile_traces=n_quantile_traces, quantile_precision=quantile_precision,
                        stats_limits=stats_limits, bar=False)

        # Assert that a survey was loaded correctly and an extra DeadTrace header was created
        assert_survey_loaded(survey, segy_path, header_index, header_cols, name, extra_headers="DeadTrace")

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
