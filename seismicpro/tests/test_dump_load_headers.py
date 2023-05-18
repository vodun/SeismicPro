"""Test methods for dump and load headers"""
# pylint: disable=too-many-arguments, redefined-outer-name

import pytest
import numpy as np

from seismicpro import SeismicDataset

from .survey import assert_surveys_equal
from .survey.conftest import survey_no_stats  # pylint: disable=unused-import
from .test_gather import compare_gathers


ARGS = [
    [["TRACE_SEQUENCE_FILE"], ["TRACE_SEQUENCE_FILE"], None, True],
    [["TRACE_SEQUENCE_FILE"], ["TRACE_SEQUENCE_FILE"], None, False],
    [["FieldRecord", "TraceNumber"], ["FieldRecord", "TraceNumber"], None, True],
    [["FieldRecord", "TraceNumber"], ["FieldRecord", "TraceNumber"], None, False],
    [["TRACE_SEQUENCE_FILE", "FieldRecord", "TraceNumber", "SourceX", "SourceY"], None, [1, 2], True],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY"], ["FieldRecord", "TraceNumber"], [0, 1], False],
]


FORMATS = [["fwf", ".", None], ["fwf", ",", None], ["csv", ".", ","], ["csv", ".", ";"]]


# pylint: disable-next=too-many-arguments
def dump_load_headers(tmp_path, container_to_dump, container_to_load, headers_to_dump, headers_to_load, format,
                      new_cols, float_precision, usecols, dump_headers_names, assert_func, dump_load_func,
                      decimal=None, sep=None, assert_kwargs=None):
    """Dump and load headers"""
    file_path = tmp_path / "tmp"

    # Generate new columns to dump. This columns will be checked after loading into new container.
    for column in new_cols:
        ranges = [[-1, 1], [-100, 100], [-1000, 1000], [0, 1], [0, 2000]]
        ix = np.random.choice(len(ranges))
        values = np.random.randint(ranges[ix][0], ranges[ix][1], size=container_to_dump.n_traces)
        if "floats" in column:
            values = values * np.random.random(container_to_dump.n_traces)
        container_to_dump.headers[column] = values

    headers_to_dump = headers_to_dump + new_cols
    if headers_to_load is not None:
        headers_to_load = headers_to_load + new_cols

    # Adding usecols for created columns if usecols is defined.
    if usecols is not None:
        usecols = usecols + [-len(new_cols) + ix for ix in range(len(new_cols))]

    kwargs = {"separator" : sep} if format == 'csv' else {"decimal": decimal}
    containers = dump_load_func(file_path=file_path, container_to_dump=container_to_dump,
                                container_to_load=container_to_load, headers_to_dump=headers_to_dump,
                                headers_to_load=headers_to_load, format=format, float_precision=float_precision,
                                usecols=usecols, dump_headers_names=dump_headers_names, decimal=decimal, sep=sep,
                                kwargs=kwargs)
    original, loaded = containers

    for column in new_cols:
        # Check that loaded floating numbers differ by no more than the number of rounded characters
        assert np.max(np.abs(original[column] - loaded[column])) <= 10**(-float_precision)
         # Avoid round and type errors
        original.headers[column] = loaded[column]
        original.headers[column] = original.headers[column].astype(loaded.headers[column].dtypes)
    assert_kwargs = {} if assert_kwargs is None else assert_kwargs
    assert_func(original, loaded, **assert_kwargs)


def dump_load_in_container(file_path, container_to_dump, container_to_load, headers_to_dump, headers_to_load, format,
                           float_precision, usecols, dump_headers_names, decimal, sep, kwargs):
    """Dump and load headers for provided containers"""
    container_to_dump.dump_headers(file_path, headers_names=headers_to_dump, dump_headers_names=dump_headers_names,
                                   float_precision=float_precision, format=format, **kwargs)
    loaded = container_to_load.load_headers(file_path, headers_names=headers_to_load, format=format,
                                            has_header=dump_headers_names, usecols=usecols, decimal=decimal, sep=sep)
    return container_to_dump, loaded


@pytest.mark.parametrize("new_cols", [["floats"], ["int"], ["floats", "int"], ["floats_1", "floats_2", "floats_3"]])
@pytest.mark.parametrize("float_precision", [5])#, 2])
@pytest.mark.parametrize("headers_to_dump,headers_to_load,usecols,dump_headers_names", ARGS)
@pytest.mark.parametrize("format,decimal,sep", FORMATS)
def test_survey_dump_load_headers(survey_no_stats, tmp_path, new_cols, float_precision, headers_to_dump,
                                  headers_to_load, usecols, dump_headers_names, format, decimal, sep):
    """Dump and load headers from survey"""
    print(format, decimal, sep)
    copy_survey = survey_no_stats.copy()
    dump_load_headers(tmp_path=tmp_path, container_to_dump=survey_no_stats, container_to_load=copy_survey,
                      headers_to_dump=headers_to_dump, headers_to_load=headers_to_load, format=format,
                      new_cols=new_cols, float_precision=float_precision, usecols=usecols,
                      dump_headers_names=dump_headers_names, assert_func=assert_surveys_equal,
                      dump_load_func=dump_load_in_container, decimal=decimal, sep=sep,
                      assert_kwargs={"ignore_dtypes": True})


@pytest.mark.parametrize("new_cols", [["floats"], ["int"], ["floats", "int"], ["floats_1", "floats_2", "floats_3"]])
@pytest.mark.parametrize("float_precision", [5, 2])
@pytest.mark.parametrize("headers_to_dump,headers_to_load,usecols,dump_headers_names", ARGS)
@pytest.mark.parametrize("format,decimal,sep", FORMATS)
def test_gather_dump_load_headers(survey_no_stats, tmp_path, new_cols, float_precision, headers_to_dump,
                                  headers_to_load, usecols, dump_headers_names, format, decimal, sep):
    """Dump and load headers from gather"""
    gather = survey_no_stats.sample_gather()
    gather_copy = gather.copy()

    dump_load_headers(tmp_path=tmp_path, container_to_dump=gather, container_to_load=gather_copy,
                      headers_to_dump=headers_to_dump, headers_to_load=headers_to_load, format=format,
                      new_cols=new_cols, float_precision=float_precision, usecols=usecols,
                      dump_headers_names=dump_headers_names, assert_func=compare_gathers,
                      dump_load_func=dump_load_in_container, decimal=decimal, sep=sep)


def dump_load_in_pipeline(container_to_dump, container_to_load, headers_to_dump, headers_to_load, file_path, format,
                           float_precision, usecols, dump_headers_names, decimal, sep, kwargs):
    """Dump and load headers from pipeline"""
    dataset = SeismicDataset(container_to_dump, container_to_load, mode='m')
    pipeline = (dataset.pipeline()
                       .load(src=[container_to_dump.name, container_to_load.name])
                       .dump_headers(src=container_to_dump.name, path=file_path, headers_names=headers_to_dump,
                                     dump_headers_names=dump_headers_names, float_precision=float_precision,
                                     format=format, **kwargs)
                       .load_headers(src=container_to_load.name, path=file_path, headers_names=headers_to_load,
                                     format=format, has_header=dump_headers_names, usecols=usecols, decimal=decimal,
                                     sep=sep)
    )
    batch = pipeline.next_batch(1)
    dumped_gather = getattr(batch, container_to_dump.name)[0]
    loaded_gather = getattr(batch, container_to_load.name)[0]
    return dumped_gather, loaded_gather


@pytest.mark.parametrize("new_cols", [["floats"], ["int"], ["floats", "int"], ["floats_1", "floats_2", "floats_3"]])
@pytest.mark.parametrize("float_precision", [5, 2])
@pytest.mark.parametrize("headers_to_dump,headers_to_load,usecols,dump_headers_names", ARGS)
@pytest.mark.parametrize("format,decimal,sep", FORMATS)
def test_pipeline_dump_load_headers(survey_no_stats, tmp_path, new_cols, float_precision, headers_to_dump,
                                    headers_to_load, usecols, dump_headers_names, format, decimal, sep):
    """Dump and load headers from pipeline and compare that resulted gathers are equal"""
    copy_survey = survey_no_stats.copy()
    copy_survey.name = survey_no_stats.name + "_copy"
    dump_load_headers(tmp_path=tmp_path, container_to_dump=survey_no_stats, container_to_load=copy_survey,
                      headers_to_dump=headers_to_dump, headers_to_load=headers_to_load, format=format,
                      new_cols=new_cols, float_precision=float_precision, usecols=usecols,
                      dump_headers_names=dump_headers_names, assert_func=compare_gathers,
                      dump_load_func=dump_load_in_pipeline, decimal=decimal, sep=sep,
                      assert_kwargs={"same_survey": False})
