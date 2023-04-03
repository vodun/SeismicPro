"""Test Survey collect_stats method"""

# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from seismicpro import Survey, make_prestack_segy

from . import assert_surveys_equal


def gen_random_traces(n_traces, n_samples):
    """Generate `n_traces` random traces."""
    return np.random.normal(size=(n_traces, n_samples)).astype(np.float32)


def gen_random_traces_some_dead(n_traces, n_samples):
    """Generate `n_traces` random traces with every third of them dead."""
    traces = np.random.uniform(size=(n_traces, n_samples)).astype(np.float32)
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
        """Return a corresponding trace from pregenerated data."""
        _ = kwargs
        return trace_data[TRACE_SEQUENCE_FILE - 1]

    path = tmp_path_factory.mktemp("stat") / "stat.sgy"
    make_prestack_segy(path, survey_size=(4, 4), origin=(0, 0), sources_step=(3, 3), receivers_step=(1, 1),
                       bin_size=(1, 1), activation_dist=(1, 1), n_samples=n_samples, sample_rate=2000, delay=0,
                       bar=False, trace_gen=gen_trace)
    return path, trace_data


class TestStats:
    """Test `collect_stats` method."""

    @pytest.mark.parametrize("init_limits", [slice(None), slice(8), slice(-4, None)])
    @pytest.mark.parametrize("n_quantile_traces", [0, 10, 100])
    @pytest.mark.parametrize("quantile_precision", [1, 2])
    @pytest.mark.parametrize("stats_limits", [None, slice(5), slice(2, 8)])
    @pytest.mark.parametrize("use_segyio_trace_loader", [True, False])
    def test_collect_stats(self, stat_segy, init_limits, n_quantile_traces, quantile_precision,
                           stats_limits, use_segyio_trace_loader):
        """Compare stats obtained by running `collect_stats` with the actual ones."""
        path, trace_data = stat_segy
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", header_cols="offset", limits=init_limits,
                        use_segyio_trace_loader=use_segyio_trace_loader, bar=False)

        survey_copy = survey.copy()
        survey.collect_stats(n_quantile_traces=n_quantile_traces, quantile_precision=quantile_precision,
                             limits=stats_limits, bar=True)

        # stats_limits take priority over init_limits
        stats_limits = init_limits if stats_limits is None else stats_limits
        trace_data = trace_data[:, stats_limits]

        # Perform basic tests of estimated quantiles since fair comparison of interpolators is complicated
        quantiles = survey.quantile_interpolator(np.linspace(0, 1, 11))
        assert np.isclose(quantiles[0], trace_data.min())
        assert np.isclose(quantiles[-1], trace_data.max())
        assert (np.diff(quantiles) >= 0).all()
        survey.quantile_interpolator = None

        # Fill the copy of the survey with actual stats and compare it with the source survey
        survey_copy.has_stats = True
        survey_copy.min = trace_data.min()
        survey_copy.max = trace_data.max()
        survey_copy.mean = trace_data.mean()
        survey_copy.std = trace_data.std()
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
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", header_cols="offset")
        survey.collect_stats()
        quantile_val = survey.get_quantile(quantile)
        assert np.isscalar(quantile) is is_scalar
        assert np.allclose(np.array(quantile_val).ravel(), survey.quantile_interpolator(quantile))

    def test_get_quantile_fails(self, stat_segy):
        """`get_quantile` must fail if survey stats were not collected."""
        path, _ = stat_segy
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", header_cols="offset")
        with pytest.raises(ValueError):
            survey.get_quantile(0.5)
