import os
import glob

import segyio
import numpy as np
from tqdm import tqdm


FILE_EXTS = ['.sgy', '.segy']


def aggregate_segys(path, out_path, recursive=False, bar=True):
    """ Path is either path to file or just to needed dir (with *.sgy or without).
    recursive goes directly into glob.glob. Is allows to add files from a nested structure via using `**`
    in path.
    """
    path = (path, ) if isinstance(path, str) else path

    files = []
    for p in path:
        # If given p is a dir name, we gonna find all files with extensions from FILE_EXTS.
        if not os.path.splitext(p)[1]:
            pattern_list = [os.path.join(p,'*'+ext) for ext in FILE_EXTS]
            concat_path = np.concatenate([glob.glob(pattern, recursive=recursive) for pattern in pattern_list])
            files.extend(concat_path)
        else:
            files.extend(glob.glob(p, recursive=recursive))

    # Check wether the files have same trace length and sample ratio and calculate number of traces
    # to allocate required buffer size for creating a new segy.
    tracecount = 0
    samples = None
    handlers_aggr = []
    for fname in files:
        segy_handler = segyio.open(fname, strict=False)
        segy_handler.mmap()
        if samples is None:
            samples = segy_handler.samples
            ext_headers = segy_handler.ext_headers
            segy_format = segy_handler.format

        if np.any(samples != segy_handler.samples):
            raise ValueError("Inconsistent samples in files!" +
                                f"Samples is {segy_handler.samples} in {fname}, previous value was {samples}")
        tracecount += segy_handler.tracecount
        handlers_aggr.append(segy_handler)

    # Create segyio spec for new file. We choose only specs that relate to unstructured data.
    spec = segyio.spec()
    spec.samples = samples
    spec.ext_headers = ext_headers
    spec.format = segy_format
    spec.tracecount = tracecount

    # Write traces and headers from `files` into new file.
    with segyio.create(out_path, spec) as to_handler:
        tr_pos = 0
        handlers_iterable = tqdm(handlers_aggr) if bar else handlers_aggr
        for from_handler in handlers_iterable:
            to_handler.trace[tr_pos: tr_pos + from_handler.tracecount] = from_handler.trace
            to_handler.header[tr_pos: tr_pos + from_handler.tracecount] = from_handler.header
            tr_pos += from_handler.tracecount
        for i in range(tracecount):
            to_handler.header[i].update({segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1})
