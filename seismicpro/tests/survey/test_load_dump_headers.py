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
def dump_load_headers(survey, tmp_path, format, new_cols, columns_to_dump, columns_to_load, usecols, dump_col_names,
                      decimal=None, sep=None):
    """Base test dump and load headers"""
    file_path = tmp_path / "tmp.csv"
    survey_copy = survey.copy()

    columns_to_dump = columns_to_dump + new_cols
    if columns_to_load is not None:
        columns_to_load = columns_to_load + new_cols
    if usecols is not None:
        usecols = usecols + [-len(new_cols) + ix for ix in range(len(new_cols))]

    for column in new_cols:
        if column == "floats":
            values = np.round(np.random.random(survey.n_traces) * 5, 2)
        else:
            values = np.random.randint(0, 10, size=survey.n_traces)
        survey.headers[column] = values

    survey.dump_headers(file_path, columns=columns_to_dump, format=format, sep=sep, dump_col_names=dump_col_names,
                        decimal=decimal)

    loaded_survey = survey_copy.load_headers(file_path, headers=columns_to_load, format=format,
                                             has_header=dump_col_names, usecols=usecols, sep=sep, decimal=decimal)
    assert_surveys_equal(survey, loaded_survey)


@pytest.mark.parametrize("decimal", ['.', ','])
@pytest.mark.parametrize("new_cols", [["float"], ["int"], ["float", "int"]])
@pytest.mark.parametrize("columns_to_dump,columns_to_load,usecols,dump_col_names", ARGS)
def test_fwf_dump_load_headers(survey, tmp_path, new_cols, columns_to_dump, columns_to_load, usecols, dump_col_names,
                               decimal):
    """Test dump and load headers in fwf format"""
    dump_load_headers(survey=survey, tmp_path=tmp_path, format="fwf", new_cols=new_cols,
                      columns_to_dump=columns_to_dump, columns_to_load=columns_to_load, usecols=usecols,
                      dump_col_names=dump_col_names, decimal=decimal)


@pytest.mark.parametrize("sep", [',', ';'])
@pytest.mark.parametrize("new_cols", [["float"], ["int"], ["float", "int"]])
@pytest.mark.parametrize("columns_to_dump,columns_to_load,usecols,dump_col_names", ARGS)
def test_csv_dump_load_headers(survey, tmp_path, new_cols, columns_to_dump, columns_to_load, usecols, dump_col_names,
                               sep):
    """Test dump and load headers in csv format"""
    dump_load_headers(survey=survey, tmp_path=tmp_path, format="csv", new_cols=new_cols,
                      columns_to_dump=columns_to_dump, columns_to_load=columns_to_load, usecols=usecols,
                      dump_col_names=dump_col_names, sep=sep)
