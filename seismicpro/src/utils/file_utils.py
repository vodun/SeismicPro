import os
import glob
from collections import namedtuple

import segyio
import numpy as np
from tqdm.auto import tqdm

from .general_utils import to_list


def aggregate_segys(in_paths, out_path, recursive=False, mmap=True, keep_exts=("sgy", "segy"), bar=True):
    in_paths = sum([glob.glob(path, recursive=recursive) for path in to_list(in_paths)], [])
    if keep_exts is not None:
        in_paths = [path for path in in_paths if os.path.splitext(path)[1][1:] in keep_exts]
    if not in_paths:
        raise ValueError("No files match given pattern")

    # Check whether all files have the same trace length and sample rate
    source_handlers = [segyio.open(path, ignore_geometry=True) for path in in_paths]
    samples = source_handlers[0].samples
    if not all(np.array_equal(samples, handler.samples) for handler in source_handlers[1:]):
        raise ValueError("Source files contain inconsistent samples")

    if mmap:
        for source_handler in source_handlers:
            source_handler.mmap()

    # Create segyio spec for the new file
    spec = segyio.spec()
    spec.samples = samples
    spec.ext_headers = source_handlers[0].ext_headers
    spec.format = source_handlers[0].format
    spec.tracecount = sum(handler.tracecount for handler in source_handlers)

    # Write traces and their headers from source files into the new one
    os.makedirs(os.path.abspath(os.path.dirname(out_path)), exist_ok=True)
    with segyio.create(out_path, spec) as out_handler:
        trace_pos = 0
        for source_handler in tqdm(source_handlers, disable=not bar):
            out_handler.trace[trace_pos : trace_pos + source_handler.tracecount] = source_handler.trace
            out_handler.header[trace_pos : trace_pos + source_handler.tracecount] = source_handler.header
            trace_pos += source_handler.tracecount
        for i in range(out_handler.tracecount):
            out_handler.header[i].update({segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1})

    # Close source segy file handlers
    for source_handler in source_handlers:
        source_handler.close()


def read_vfunc(path):
    vfunc_list = []
    VFUNC = namedtuple("VFUNC", ["inline", "crossline", "x", "y"])
    with open(path) as file:
        for data in file.read().split("VFUNC")[1:]:
            data = data.split()
            inline, crossline = int(data[0]), int(data[1])
            data = np.array(data[2:], dtype=np.float64)
            if len(data) % 2 != 0:
                raise ValueError("Data length for each VFUNC record must be even")
            VFUNC(inline, crossline, data[::2], data[1::2])
            vfunc_list.append(VFUNC(inline, crossline, data[::2], data[1::2]))
    return vfunc_list


def read_single_vfunc(path):
    file_data = read_vfunc(path)
    if len(file_data) != 1:
        raise ValueError(f"Input file must contain a single vfunc, but {len(file_data)} were found in {path}")
    return file_data[0]
