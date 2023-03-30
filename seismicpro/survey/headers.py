"""SEG-Y trace headers loading routines"""

import os
import mmap
from struct import unpack
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import segyio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..const import TRACE_HEADER_SIZE, ENDIANNESS
from ..utils import ForPoolExecutor


def define_unpacking_format(headers_to_load):
    """Return a format string to unpack `headers_to_load` trace headers from a byte sequence of header values.

    The string encodes each trace header in the order they are stored in a SEG-Y file with one of the following
    characters:
    * "h" - if the header is being loaded and its value is stored as a 16-bit integer,
    * "i" - if the header is being loaded and its value is stored as a 32-bit integer,
    * "xx" - if the header is not being loaded and its value is stored as a 16-bit integer,
    * "xxxx" - if the header is not being loaded and its value is stored as a 32-bit integer.
    """
    header_to_byte = segyio.tracefield.keys
    byte_to_header = {val: key for key, val in header_to_byte.items()}
    start_bytes = sorted(header_to_byte.values())
    byte_to_len = {start: end - start for start, end in zip(start_bytes, start_bytes[1:] + [TRACE_HEADER_SIZE + 1])}
    len_to_code = {2: "h", 4: "i"}  # Each header value is represented either by int16 or int32

    headers_to_load_bytes = {header_to_byte[header] for header in headers_to_load}
    headers_to_code = {byte: len_to_code[header_len] if byte in headers_to_load_bytes else "x" * header_len
                       for byte, header_len in byte_to_len.items()}
    headers_format = "".join(headers_to_code[byte] for byte in start_bytes)
    headers_order = [byte_to_header[byte] for byte in sorted(headers_to_load_bytes)]
    return headers_format, headers_order


def read_headers_chunk(path, chunk_offset, chunk_size, trace_stride, headers_format, endian):
    """Read `chunk_size` trace headers starting from `chunk_offset` byte in the SEG-Y file.

    Headers to load are described by `headers_format` format string as described in :func:`~.define_unpacking_format`.
    The function returns an `np.ndarray` of loaded headers values with shape `(chunk_size, n_loaded_headers)`.
    """
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            headers = np.ndarray(buffer=mm, dtype=np.dtype(f"V{TRACE_HEADER_SIZE}"), offset=chunk_offset,
                                 shape=chunk_size, strides=trace_stride).tolist()
    # Construct a format string for the whole chunk
    chunk_format_str = ENDIANNESS[endian] + headers_format * chunk_size
    # Unpack headers and cast them to int32 since their values are at most 4-byte integers according to SEG-Y spec
    return np.array(unpack(chunk_format_str, b"".join(headers)), dtype=np.int32).reshape(chunk_size, -1)


def load_headers(path, headers_to_load, trace_data_offset, trace_size, n_traces, endian, chunk_size, n_workers, bar):
    """Load `headers_to_load` trace headers from a SEG-Y file for each trace as a `pd.DataFrame`.

    Headers values are loaded in parallel processes in chunks of size no more than `chunk_size`. The algorithm first
    loads all headers for each trace and then keeps only the requested ones since this approach is faster than
    consequent seeks and reads.
    """
    # Split the whole file into chunks no larger than chunk_size
    trace_stride = TRACE_HEADER_SIZE + trace_size
    n_chunks, last_chunk_size = divmod(n_traces, chunk_size)
    chunk_sizes = [chunk_size] * n_chunks
    if last_chunk_size:
        chunk_sizes += [last_chunk_size]
    chunk_starts = np.cumsum([0] + chunk_sizes[:-1])
    chunk_offsets = trace_data_offset + chunk_starts * trace_stride

    # Construct a format string to unpack trace headers from a byte sequence and preallocate headers buffer
    headers_format, headers_order = define_unpacking_format(headers_to_load)
    headers = np.empty((n_traces, len(headers_to_load)), dtype=np.int32)

    # Process passed n_workers and select an appropriate pool executor
    if n_workers is None:
        n_workers = os.cpu_count()
    n_workers = min(len(chunk_sizes), n_workers)
    executor_class = ForPoolExecutor if n_workers == 1 else ProcessPoolExecutor

    # Load trace headers for each chunk
    with tqdm(total=n_traces, desc="Trace headers loaded", disable=not bar) as pbar:
        with executor_class(max_workers=n_workers) as pool:
            def callback(future, start_pos):
                chunk_headers = future.result()
                n_headers = len(chunk_headers)
                headers[start_pos : start_pos + n_headers] = chunk_headers
                pbar.update(n_headers)

            for start, size, offset in zip(chunk_starts, chunk_sizes, chunk_offsets):
                future = pool.submit(read_headers_chunk, path, offset, size, trace_stride, headers_format, endian)
                future.add_done_callback(partial(callback, start_pos=start))

    return pd.DataFrame(headers, columns=headers_order)
