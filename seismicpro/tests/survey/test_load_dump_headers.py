"""Test Survey methods for dump and load headers"""

import pytest
import numpy as np

from . import assert_surveys_equal


ARGS = [
    [["TRACE_SEQUENCE_FILE"], ["TRACE_SEQUENCE_FILE"], None, True],
    [["TRACE_SEQUENCE_FILE"], ["TRACE_SEQUENCE_FILE"], None, False],
    [["FieldRecord", "TraceNumber"], ["FieldRecord", "TraceNumber"], None, True],
    [["FieldRecord", "TraceNumber"], ["FieldRecord", "TraceNumber"], None, False],
    [["TRACE_SEQUENCE_FILE", "FieldRecord", "TraceNumber", "SourceX", "SourceY"], None, [1, 2], True],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY"], ["FieldRecord", "TraceNumber"], [0, 1], False],
]


# TODO: addtests for keep_all_headers, skiprows and join_on_headers
def load_dump_headers(survey, tmp_path, format, new_cols, float_precision, headers_to_dump, headers_to_load, usecols,
                      dump_headers_names, decimal=None, sep=None):
    """Base test dump and load headers"""
    file_path = tmp_path / "tmp.csv"
    survey_copy = survey.copy()

    headers_to_dump = headers_to_dump + new_cols
    if headers_to_load is not None:
        headers_to_load = headers_to_load + new_cols
    if usecols is not None:
        usecols = usecols + [-len(new_cols) + ix for ix in range(len(new_cols))]

    for column in new_cols:
        if column == "floats":
            values = np.round(np.random.random(survey.n_traces) * 5, 3)
        else:
            values = np.random.randint(0, 10, size=survey.n_traces)
        survey.headers[column] = values

    kwargs = {"separator" : sep} if format == 'csv' else {"decimal": decimal}
    survey.dump_headers(file_path, headers=headers_to_dump, dump_headers_names=dump_headers_names,
                        float_precision=float_precision, format=format, **kwargs)

    loaded_survey = survey_copy.load_headers(file_path, headers=headers_to_load, format=format,
                                             has_header=dump_headers_names, usecols=usecols, sep=sep, decimal=decimal)
    assert_surveys_equal(survey, loaded_survey)


@pytest.mark.parametrize("decimal", ['.', ','])
@pytest.mark.parametrize("float_precision", [None, 2])
@pytest.mark.parametrize("new_cols", [["float"], ["int"], ["float", "int"]])
@pytest.mark.parametrize("headers_to_dump,headers_to_load,usecols,dump_headers_names", ARGS)
def test_fwf_load_dump_headers(survey, tmp_path, new_cols, float_precision, headers_to_dump, headers_to_load, usecols,
                               dump_headers_names, decimal):
    """Test dump and load headers in fwf format"""
    load_dump_headers(survey=survey, tmp_path=tmp_path, format="fwf", float_precision=float_precision,
                      new_cols=new_cols, headers_to_dump=headers_to_dump, headers_to_load=headers_to_load,
                      usecols=usecols, dump_headers_names=dump_headers_names, decimal=decimal)


@pytest.mark.parametrize("sep", [',', ';'])
@pytest.mark.parametrize("float_precision", [None, 2])
@pytest.mark.parametrize("new_cols", [["float"], ["int"], ["float", "int"]])
@pytest.mark.parametrize("headers_to_dump,headers_to_load,usecols,dump_headers_names", ARGS)
def test_csv_load_dump_headers(survey, tmp_path, new_cols, float_precision, headers_to_dump, headers_to_load, usecols,
                               dump_headers_names, sep):
    """Test dump and load headers in csv format"""
    load_dump_headers(survey=survey, tmp_path=tmp_path, format="csv",  float_precision=float_precision,
                      new_cols=new_cols, headers_to_dump=headers_to_dump, headers_to_load=headers_to_load,
                      usecols=usecols, dump_headers_names=dump_headers_names, sep=sep)
