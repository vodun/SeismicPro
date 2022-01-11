"""Test Survey collect_stats and dead_traces-related methods"""

# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from seismicpro import Survey, make_prestack_segy
from seismicpro.src.custom_headers import HDR_DEAD_TRACE

from . import assert_surveys_equal, assert_survey_processed_inplace


def gen_random_traces(n_traces, n_samples):
    """Generate `n_traces` random traces."""
    return np.random.normal(size=(n_traces, n_samples))


def gen_random_traces_some_dead(n_traces, n_samples):
    """Generate `n_traces` random traces with every third of them dead."""
    traces = np.random.uniform(size=(n_traces, n_samples))
    traces[::3] = 0
    return traces


@pytest.fixture(scope="module", params=[gen_random_traces, gen_random_traces_some_dead])
def stat_segy(tmp_path_factory, request):
    """Return a path to a SEG-Y file and its trace data to estimate its statistics."""
    n_traces = 16
    n_samples = 10
    trace_gen = request.param
    trace_data = trace_gen(n_traces, n_samples)

    def gen_trace(TRACE_SEQUENCE_FILE, **kwargs):  # pylint: disable=invalid-name
        """Return a corresponding trace from a pregenerated data."""
        _ = kwargs
        return trace_data[TRACE_SEQUENCE_FILE - 1]

    path = tmp_path_factory.mktemp("stat") / "stat.sgy"
    make_prestack_segy(path, survey_size=(4, 4), origin=(0, 0), sources_step=(3, 3), recievers_step=(1, 1),
                        bin_size=(1, 1), activation_dist=(1, 1), n_samples=n_samples, sample_rate=2000, delay=0,
                        bar=False, trace_gen=gen_trace)
    return path, trace_data


class TestStats:
    """Test `collect_stats` and `remove_dead_traces` methods."""

    @pytest.mark.parametrize("limits", [slice(8), slice(-4, None)])
    @pytest.mark.parametrize("n_quantile_traces", [0, 10, 100])
    @pytest.mark.parametrize("quantile_precision", [1, 2])
    @pytest.mark.parametrize("stats_limits", [None, slice(5), slice(2, 8)])
    def test_collect_stats(self, stat_segy, limits, n_quantile_traces, quantile_precision, stats_limits):
        """Compare stats obtained by running `collect_stats` with the actual ones."""
        path, trace_data = stat_segy
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", header_cols="offset", limits=limits,
                        collect_stats=False)
        survey_copy = survey.copy()
        survey.collect_stats(n_quantile_traces=n_quantile_traces, quantile_precision=quantile_precision,
                             stats_limits=stats_limits, bar=True)

        # stats_limits take priority over survey limits
        stats_limits = limits if stats_limits is None else stats_limits
        trace_data = trace_data[:, stats_limits]
        is_dead = np.isclose(trace_data.min(axis=1), trace_data.max(axis=1))
        non_dead_traces = trace_data[~is_dead].ravel()

        # Perform basic tests of estimated quantiles since fair comparison of interpolators is complicated
        quantiles = survey.quantile_interpolator(np.linspace(0, 1, 11))
        assert np.isclose(quantiles[0], non_dead_traces.min())
        assert np.isclose(quantiles[-1], non_dead_traces.max())
        assert (np.diff(quantiles) >= 0).all()
        survey.quantile_interpolator = None

        # Fill the copy of the survey with actual stats and compare it with the source survey
        survey_copy.n_dead_traces = is_dead.sum()
        survey_copy.has_stats = True
        survey_copy.min = non_dead_traces.min()
        survey_copy.max = non_dead_traces.max()
        survey_copy.mean = non_dead_traces.mean()
        survey_copy.std = non_dead_traces.std()
        assert_surveys_equal(survey, survey_copy)


    @pytest.mark.parametrize("quantile, is_scalar", [
        [0.5, True],
        [0, True],
        [1, True],
        [[0.05, 0.95], False],
        [[0.3, 0.3], False]
    ])
    def test_get_quantile(self, stat_segy, quantile, is_scalar):
        """Run `get_quantile` and check the returned value and its type."""
        path, _ = stat_segy
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", header_cols="offset", collect_stats=True, bar=False)
        quantile_val = survey.get_quantile(quantile)
        assert np.isscalar(quantile) is is_scalar
        assert np.allclose(np.array(quantile_val).ravel(), survey.quantile_interpolator(quantile))

    def test_get_quantile_fails(self, stat_segy):
        """`get_quantile` must fail if survey stats were not collected."""
        path, _ = stat_segy
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", header_cols="offset")
        with pytest.raises(ValueError):
            survey.get_quantile(0.5)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("pre_mark_dead", [True, False])
    def test_remove_dead_traces(self, stat_segy, inplace, pre_mark_dead):
        """Check that `remove_dead_traces` properly updates survey `headers` and sets `n_dead_traces` counter to 0."""

        path, trace_data = stat_segy
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", header_cols="offset")
        survey_copy = survey.copy()

        if pre_mark_dead:
            survey.mark_dead_traces()

        survey_filtered = survey.remove_dead_traces(inplace=inplace)

        is_dead = np.isclose(trace_data.min(axis=1), trace_data.max(axis=1))
        survey_copy.headers = survey_copy.headers.loc[~is_dead]
        survey_copy.n_dead_traces = 0
        survey_copy.headers[HDR_DEAD_TRACE] = False

        # Validate that dead traces are not present
        assert survey_filtered.n_dead_traces == 0
        assert survey_filtered.headers.index.is_monotonic_increasing
        assert_surveys_equal(survey_filtered, survey_copy)
        assert_survey_processed_inplace(survey, survey_filtered, inplace)

    def test_mark_dead_traces(self, stat_segy):
        """Check that `mark_dead_traces` properly updates survey `headers` and sets `n_dead_traces` counter to 0."""

        path, trace_data = stat_segy
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", header_cols="offset")
        survey_copy = survey.copy()

        survey.mark_dead_traces()

        is_dead = np.isclose(trace_data.min(axis=1), trace_data.max(axis=1))
        survey_copy.headers[HDR_DEAD_TRACE] = False
        survey_copy.headers.loc[is_dead, HDR_DEAD_TRACE] = True
        survey_copy.n_dead_traces = np.sum(is_dead)

        assert_surveys_equal(survey, survey_copy)
