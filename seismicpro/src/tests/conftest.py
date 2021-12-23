"""Generate a test SEG-Y file"""

import pytest

from seismicpro import make_prestack_segy


@pytest.fixture(scope="package", autouse=True)
def segy_path(tmp_path_factory):
    """Create a temporary SEG-Y file with randomly generated traces."""
    path = tmp_path_factory.mktemp("data") / "test_prestack.sgy"
    make_prestack_segy(path, survey_size=(300, 300), origin=(0, 0), sources_step=(50, 150), recievers_step=(100, 25),
                       bin_size=(50, 50), activation_dist=(200, 200), n_samples=1000, sample_rate=2000, delay=0,
                       bar=False, trace_gen=None)
    return path
