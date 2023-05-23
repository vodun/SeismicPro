import pytest
import numpy as np
import pandas as pd

from seismicpro import Survey
from seismicpro.utils import dump_dataframe, load_dataframe


def add_columns(container, size):
    def _gen_col(vmin, vmax, dtype, size):
        int_part = np.random.randint(vmin, vmax, size=size)
        if dtype == "float":
            return int_part * np.random.random(size=size)
        return int_part

    ranges = [[0, 1], [-1, 1], [0, 1000], [-1000, 1000], [-1000000, 1000000]]
    for vmin, vmax in ranges:
        for dtype in ["int", "float"]:
            container[f"{dtype}_{np.abs(vmin)}_{vmax}"] = _gen_col(vmin, vmax, dtype, size)
    return container


@pytest.fixture(params=["survey", "gather"], scope="module")
def containers(segy_path, request):
    survey = Survey(segy_path, header_index="FieldRecord", header_cols="all", n_workers=1, bar=False, validate=False)
    container = survey.sample_gather() if request.param == "gather" else survey
    container_with_cols = add_columns(container.copy(), container.n_traces)
    return container, container_with_cols


@pytest.fixture(scope="module")
def dataframe():
    df_dict = {}
    return pd.DataFrame(add_columns(df_dict, 100))


@pytest.mark.parametrize("headers_to_dump,headers_to_load,usecols,has_header,float_precision,decimal,sep", [
    [["int_0_1000"], ["int_0_1000"], None, True, 2, ".", ","],
    [["float_1_1"], ["float_1_1"], None, False, 3, ".", ";"],
    [["int_0_1000", "float_0_1000", "int_1000_1000", "float_1000_1000"],
     ["int_0_1000", "float_0_1000", "int_1000_1000", "float_1000_1000"], None, True, 3, ",", ","],
    [["int_0_1000", "float_0_1000", "int_1000_1000", "float_1000_1000"], None, [1, 2], True, 5, ".", ","],
    [["int_0_1000", "float_0_1000", "int_1000_1000", "float_1000_1000"], ["float_0_1000", "int_1000_1000"], [1, 2],
     False, 5, ",", ";"],
    [["float_0_1", "float_1_1", "int_1000000_1000000", "float_1000000_1000000"],
     ["float_1_1", "float_1000000_1000000"], None, True, 3, ".", ";"],
])
@pytest.mark.parametrize("format", ["fwf", "csv"])
def test_dump_load_dataframe(tmp_path, dataframe, headers_to_dump, headers_to_load, usecols, has_header,
                             float_precision, decimal, sep, format):
    file_path = tmp_path / "tmp"

    kwargs = {"decimal": decimal} if format == "fwf" else {"separator": sep}
    dump_dataframe(file_path, dataframe[headers_to_dump], has_header=has_header, format=format,
                   float_precision=float_precision, **kwargs)

    kwargs = kwargs if format == "fwf" else {"sep": sep}
    loaded_df = load_dataframe(file_path, columns=headers_to_load, has_header=has_header, usecols=usecols,
                               format=format, **kwargs)
    assert_headers = headers_to_load
    if headers_to_load is None:
        assert_headers = headers_to_dump if usecols is None else np.array(headers_to_dump)[usecols]
    assert ((dataframe[assert_headers] - loaded_df).max() <= 10**(-float_precision)).all()


@pytest.mark.parametrize("headers_to_dump,headers_to_load,usecols,has_header,float_precision,decimal,sep",[
    [["TRACE_SEQUENCE_FILE", "int_0_1000"], ["TRACE_SEQUENCE_FILE", "int_0_1000"], None, True, 2, ".", ","],
    [["TRACE_SEQUENCE_FILE", "float_1000_1000"], ["TRACE_SEQUENCE_FILE", "float_1000_1000"], None, False, 2, ",", ","],
    [["FieldRecord", "TraceNumber", "int_1000_1000", "float_1_1", "float_0_1000"],
     ["FieldRecord", "TraceNumber", "int_1000_1000", "float_1_1", "float_0_1000"], None, True, 4, ".", ";"],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY", "float_1000_1000"],
     ["FieldRecord", "TraceNumber", "float_1000_1000"], [0, 1, -1], False, 2, ",", ";"],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY", "float_1000_1000"], None, [0, 1, -1], True, 2, ".", ","],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY", "int_1000000_1000000", "float_1000000_1000000"],
     None, None, True, 2, ".", ","],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY", "int_0_1", "float_0_1", "float_1000_1000"],
     ["FieldRecord", "TraceNumber", "int_0_1", "float_0_1", "float_1000_1000"], None, True, 2, ",", ","]
])
@pytest.mark.parametrize("format", ["fwf", "csv"])
def test_dump_container(tmp_path, containers, headers_to_dump, headers_to_load, usecols, has_header, float_precision,
                        decimal, sep, format):
    file_path = tmp_path / "tmp"
    _, container_to_dump = containers

    kwargs = {"decimal": decimal} if format == "fwf" else {"separator": sep}
    container_to_dump.dump_headers(file_path, headers_names=headers_to_dump, dump_headers_names=has_header,
                                   format=format, float_precision=float_precision, **kwargs)

    kwargs = kwargs if format == "fwf" else {"sep": sep}
    loaded_df = load_dataframe(file_path, columns=headers_to_load, has_header=has_header, usecols=usecols,
                               format=format, **kwargs)

    assert_headers = headers_to_load
    if headers_to_load is None:
        assert_headers = headers_to_dump if usecols is None else np.array(headers_to_dump)[usecols]
    headers = container_to_dump.get_headers(assert_headers)
    assert ((headers - loaded_df).max() <= 10**(-float_precision)).all()
