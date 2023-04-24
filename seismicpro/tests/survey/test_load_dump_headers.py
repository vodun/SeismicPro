"""Test Survey methods for dump and load headers"""
import os

import pytest
import numpy as np

from seismicpro.utils import to_list
from . import assert_surveys_equal

@pytest.mark.parametrize('names', ["float_point", ["float_point", "float_point"]])
@pytest.mark.parametrize('join_on_headers', ["TRACE_SEQUENCE_FILE", ["FieldRecord", "TraceNumber"]])
@pytest.mark.parametrize('format', ["fwf", "csv"])
@pytest.mark.parametrize('dump_col_names', [True, False])
def test_dump_load_headers(survey, tmp_path, names, join_on_headers, format, dump_col_names):
    """Test dump and load headers"""
    file_path = os.path.join(tmp_path, "tmp.csv")
    dump_cols = []
    survey_copy = survey.copy()
    for ix in range(len(to_list(names))):
        point_col = f"float_point_{ix}"
        survey[point_col] = np.round(np.random.random(survey.n_traces)*10, decimals=4)
        dump_cols.append(point_col)

    columns = to_list(join_on_headers) + dump_cols
    survey.dump_headers(file_path, columns=columns, format=format, dump_col_names=dump_col_names)

    loaded_survey = survey_copy.load_headers(file_path, headers=columns, join_on_headers=join_on_headers,
                                             format=format, has_header=dump_col_names)
    assert_surveys_equal(survey, loaded_survey)
