"""Test Survey methods for dump and load headers"""
import os

import pytest
import numpy as np

from seismicpro.utils import to_list
from . import assert_surveys_equal

@pytest.mark.parametrize('names', ["float_comma", "float_point", ["float_point", "float_point"]])
@pytest.mark.parametrize('index_col', ["TRACE_SEQUENCE_FILE", ["FieldRecord", "TraceNumber"]])
@pytest.mark.parametrize('format', ["fwf", "csv"])
@pytest.mark.parametrize('dump_col_names', [True, False])
def test_dump_load_headers(survey, tmp_path, names, index_col, format, dump_col_names):
    """Test dump and load headers"""
    file_path = os.path.join(tmp_path, "tmp.csv")
    dump_cols = []
    print(names)
    survey_copy = survey.copy()
    for ix, name in enumerate(to_list(names)):
        if "comma" in name:
            comma_col = f"float_comma_{ix}"
            float_values = (10 * np.random.random(survey.n_traces)).astype('str')
            survey[comma_col] = list(map(lambda s: s.replace(".", ","), float_values))
            decimal = ","
            dump_cols.append(comma_col)
        elif "point" in name:
            point_col = f"float_point_{ix}"
            survey[point_col] = np.round(np.random.random(survey.n_traces)*10, decimals=4)
            decimal = "."
            dump_cols.append(point_col)
        else:
            dump_cols.append(name)

    columns = to_list(index_col) + dump_cols
    survey.dump_headers(file_path, columns=columns, format=format, dump_col_names=dump_col_names)

    skiprows = 1 if dump_col_names else None
    loaded_survey = survey_copy.load_headers(file_path, names=columns, index_col=index_col, format=format,
                                             skiprows=skiprows, decimal=decimal)

    for name in to_list(columns):
        if "comma" in name:
            loaded_floats = loaded_survey[name]
            assert isinstance(loaded_floats[0], np.floating)
            assert np.allclose(float_values.astype(np.float64), loaded_floats)
            survey.headers.drop(columns=name, inplace=True)
            loaded_survey.headers.drop(columns=name, inplace=True)

        if "point" in name:
            loaded_floats = loaded_survey[name]
            assert isinstance(loaded_floats[0], np.floating)
            assert np.allclose(loaded_floats, survey[name])
            survey.headers.drop(columns=name, inplace=True)
            loaded_survey.headers.drop(columns=name, inplace=True)

    assert_surveys_equal(survey, loaded_survey)
