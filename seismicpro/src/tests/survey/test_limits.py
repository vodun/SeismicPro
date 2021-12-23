"""Test Survey.set_limits method"""

import pytest

from . import assert_survey_limits


LIMITS = [
    # None equals to loading of whole traces
    (None, slice(0, 1000, 1)),

    # Ints and tuples are converted to a corresponding slice
    (10, slice(0, 10, 1)),
    ((100, 200), slice(100, 200, 1)),
    ((100, 500, 5), slice(100, 500, 5)),
    ((None, 200, 3), slice(0, 200, 3)),

    # Slices with positive attributes are passed as-is
    (slice(700, 800), slice(700, 800, 1)),
    (slice(400, None, 4), slice(400, 1000, 4)),

    # Handle negative bounds (note that that each trace has 1000 samples)
    (-100, slice(0, 900, 1)),
    (slice(0, -100), slice(0, 900, 1)),
    (slice(-200, -100), slice(800, 900, 1)),
    (slice(-200), slice(0, 800, 1)),
]


@pytest.mark.parametrize("limits, slice_limits", LIMITS)
def test_set_limits(survey, limits, slice_limits):
    survey.set_limits(limits)
    assert_survey_limits(survey, slice_limits)


FAIL_LIMITS = [
    # Negetive step is not allowed
    (200, 100, -2),
    slice(-100, -500, -1),

    # Slicing must not return empty traces
    slice(-100, -200),
    slice(500, 100),
]


@pytest.mark.parametrize("limits", FAIL_LIMITS)
def test_set_limits_fails(survey, limits):
    with pytest.raises(ValueError):
        survey.set_limits(limits)
