"""SEG-Y trace headers loading routines"""

import os
import mmap
import warnings
from struct import unpack
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import segyio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..const import TRACE_HEADER_SIZE, ENDIANNESS
from ..utils import ForPoolExecutor, IDWInterpolator


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

def validate_headers(headers, full=False):
    msg_list = []
    n_traces = headers.shape[0]

    shot_cols = ["SourceX", "SourceY"]
    rec_cols = ["GroupX", "GroupY"]
    cdp_cols = ["CDP_X", "CDP_Y"]
    bin_cols = ["INLINE_3D", "CROSSLINE_3D"]

    loaded_columns = headers.columns.values
    avalible_columns = set(loaded_columns[headers.any(axis=0)])

    zero_columns = set(loaded_columns) - set(avalible_columns)
    if zero_columns:
        msg_list.append("Empty columns: " + ", ".join(zero_columns))

    has_full_duplicates = headers.duplicated().any() if full else False
    if has_full_duplicates:
        msg_list.append(f"Duplicate traces")

    if {"FieldRecord", "TraceNumber"} <= avalible_columns:
        n_unique_ids = (~headers[["FieldRecord", "TraceNumber"]].duplicated()).sum()
        has_unique_trace_id = n_unique_ids == n_traces
        if not has_unique_trace_id:
            msg_list.append("Non-unique traces indentifier (FieldRecord, TraceNumber)")

    if {"FieldRecord", *shot_cols} <= avalible_columns:
        fr_with_coords = headers[["FieldRecord", *shot_cols]].drop_duplicates()
        n_unique_fr = fr_with_coords["FieldRecord"].nunique()
        has_unique_coords = len(fr_with_coords) == n_unique_fr
        if not has_unique_coords:
            msg_list.append("Several pairs of coordinates (SourceX, SourceY) for single FieldRecond")

    # Check that Eqlidian distance calculated based on the coords from shot to receiver is close to the one stored
    # in trace headers
    if {*shot_cols, *rec_cols, "offset"} <= avalible_columns:
        calculated_offsets = np.sqrt(np.sum((headers[shot_cols].values - headers[rec_cols].values)**2, axis=1))
        # Avoiding small offsets since they may leads to false positive estimation.
        mask = headers["offset"] > 20
        real_offsets = headers["offset"].values[mask]
        has_correct_offsets = np.all(np.abs(calculated_offsets[mask] - real_offsets) / real_offsets < 0.1)
        if not has_correct_offsets:
            msg_list.append("Mismatch of the distance between shots and receivers posinitons for each"
                            "\n    trace and offsets in headers")

    if {*cdp_cols, *bin_cols} <= avalible_columns:
        unique_inline_cdp = headers[[*bin_cols, *cdp_cols]].drop_duplicates()
        unique_cdp = unique_inline_cdp[cdp_cols].drop_duplicates()
        has_unique_inline_to_cdp = len(unique_inline_cdp) == len(unique_cdp)
        if not has_unique_inline_to_cdp:
            msg_list.append("Non-unique mapping of geographic (CDP_X, CDP_Y) to binary(INLINE_3D/"
                            "\n    CROSSLINE_3D) coordinates")

    if {*shot_cols, *rec_cols, *cdp_cols} <= avalible_columns:
        raw_cdp = (headers[shot_cols].values + headers[rec_cols].values) / 2
        has_consistent_geo_coords = np.all((raw_cdp - headers[cdp_cols].values) / headers[cdp_cols].values < 0.1)
        if not has_consistent_geo_coords:
            msg_list.append("Unconsistent range of some geographic coordinates (SourceX, SourceY, GroupX,"
                            "\n    GroupY, CDP_X, CDP_Y)")

    if {*shot_cols, "SourceSurfaceElevation"} <= avalible_columns:
        ushot_elev_coords = headers[[*shot_cols, "SourceSurfaceElevation"]].drop_duplicates()
        ushot_coords = ushot_elev_coords[shot_cols].drop_duplicates()
        has_shot_uniq_elevs = len(ushot_elev_coords) == len(ushot_coords)
        if not has_shot_uniq_elevs:
            msg_list.append("Different surface elevation (SourceSurfaceElevation) for at least one shot")

    if {*rec_cols, "ReceiverGroupElevation"} <= avalible_columns:
        urec_elev_coords = headers[[*rec_cols, "ReceiverGroupElevation"]].drop_duplicates()
        urec_coords = urec_elev_coords[rec_cols].drop_duplicates()
        has_rec_uniq_elevs = len(urec_elev_coords) == len(urec_coords)
        if not has_rec_uniq_elevs:
            msg_list.append("Different surface elevation (ReceiverGroupElevation) for at least"
                            "\n    one receiver")

    if {*shot_cols, *rec_cols, "ReceiverGroupElevation", "SourceSurfaceElevation"} <= avalible_columns:
        shot_elevations = ushot_elev_coords.values[:, 2]
        shot_interp = IDWInterpolator(ushot_elev_coords.values[:, :2], shot_elevations, neighbors=3)
        mask = shot_elevations > 0
        rec_by_shot = np.abs(shot_interp(ushot_elev_coords.values[:, :2][mask]) - shot_elevations[mask])
        has_correct_elevations = np.all(rec_by_shot / shot_elevations[mask] < 0.1)
        if not has_correct_elevations:
            msg_list.append("Unconsistent values in elevations-related headers (ReceiverGroupElevation,"
                            "\n    SourceSurfaceElevation)")

    if len(msg_list) > 0:
        line = "\n\n" + "-"*80
        msg = line +  f"\n\nThe loaded Survey has the following problems with trace headers:"
        msg = f"{msg}" + "".join([f"\n\n {i+1}. {msg}" for i, msg in zip(range(len(msg_list)), msg_list)])
        msg += line
        warnings.warn(msg, Warning)
