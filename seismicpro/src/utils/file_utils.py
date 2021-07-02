"""Implements functions to load and dump data in various formats"""

import os
import glob
from collections import namedtuple

import segyio
import numpy as np
from tqdm.auto import tqdm

from .general_utils import to_list


def aggregate_segys(in_paths, out_path, recursive=False, mmap=True, keep_exts=("sgy", "segy"), bar=True):
    """Merge several segy files into a single one.

    Parameters
    ----------
    in_paths : str or list of str
        Glob mask or masks to search for source files to merge.
    out_path : str
        A path to the resulting merged file.
    recursive : bool, optional, defaults to False
        Whether to treat '**' pattern as zero or more directories to perfrom a recursive file search.
    mmap : bool, optional, defaults to True
        Whether to perform memory mapping of input files. Setting this flag to `True` may result in faster reads.
    keep_exts : None, array-like, optional, defaults to ("sgy", "segy")
        Extensions of files to use for merging. If `None`, no filtering is performed.
    bar : bool, optional, defaults to True
        Whether to show the progres bar.

    Raises
    ------
    ValueError
        If no files match the given pattern.
        If source files contain inconsistent samples.
    """
    in_paths = sum([glob.glob(path, recursive=recursive) for path in to_list(in_paths)], [])
    if keep_exts is not None:
        in_paths = [path for path in in_paths if os.path.splitext(path)[1][1:] in keep_exts]
    if not in_paths:
        raise ValueError("No files match the given pattern")

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
    """Read a file with vertical functions in Paradigm Echos VFUNC format.

    The file may have one or more records with the following structure:
    VFUNC [inline] [crossline]
    [x1] [y1] [x2] [y2] ... [xn] [yn]

    Parameters
    ----------
    path : str
        A path to the file.

    Returns
    -------
    vfunc_list : list of namedtuples
        List of loaded vertical functions. Each vfunc is a `namedtuple` with the following fields: `inline`,
        `crossline`, `x` and `y`, where `x` and `y` are 1d np.ndarrays with the same length.

    Raises
    ------
    ValueError
        If data length for any VFUNC record is odd.
    """
    vfunc_list = []
    VFUNC = namedtuple("VFUNC", ["inline", "crossline", "x", "y"])
    with open(path) as file:
        for data in file.read().split("VFUNC")[1:]:
            data = data.split()
            inline, crossline = int(data[0]), int(data[1])
            data = np.array(data[2:], dtype=np.float64)
            if len(data) % 2 != 0:
                raise ValueError("Data length for each VFUNC record must be even")
            vfunc_list.append(VFUNC(inline, crossline, data[::2], data[1::2]))
    return vfunc_list


def read_single_vfunc(path):
    """Read a single vertical function from a file in Paradigm Echos VFUNC format.

    The file must have exactly one record with the following structure:
    VFUNC [inline] [crossline]
    [x1] [y1] [x2] [y2] ... [xn] [yn]

    Parameters
    ----------
    path : str
        A path to the file.

    Returns
    -------
    vfunc : namedtuple
        Vertical function with the following fields: `inline`, `crossline`, `x` and `y`, where `x` and `y` are 1d
        np.ndarrays with the same length.

    Raises
    ------
    ValueError
        If data length for any VFUNC record is odd.
        If the file does not contain a single vfunc.
    """
    file_data = read_vfunc(path)
    if len(file_data) != 1:
        raise ValueError(f"Input file must contain a single vfunc, but {len(file_data)} were found in {path}")
    return file_data[0]


def make_prestack_segy(path_segy, sources_size=500, activation_dist=500, bin_size=50, samples=1500, trace_gen=None,
                       dist_source_lines=300, dist_sources=50, dist_reciever_lines=100, dist_recievers=25, **kwargs):
    # pylint: disable=invalid-name
    """ Makes a prestack segy with square geometry. Segy headers are filled with calculated values.

    Parameters
    ----------
    path_segy : str
        Path to store new segy.
    trace_gen : callable or None
        ...
    survey_size : int
        ...
    activation_dist : int
        ...
    bin_size : int
        ...
    samples : int
        ...
    dist_source_lines : int
        ...
    dist_sources : int
        ...
    dist_reciever_lines : int
        ...
    dist_recievers : int
        ...
    kwargs : dict
        format : int
            floating-point mode. 5 stands for IEEE-floating point, which is the standard -
            it is set as the default.
        sample_rate : int
            sampling frequency of the seismic in microseconds. Most commonly is equal to 2000
            microseconds for on-land seismic.
        delay : int
            delay time of the seismic in microseconds. The default is 0.
    """
    # By default traces are filled with random noise
    if trace_gen is None:
        def trace_gen(**kwargs):
            samples = kwargs.get('samples')
            return np.random.normal(size=samples).astype(np.float32)

    def generate_coordinates(start, stop, dist, dist_lines, lines_axis=0):
        """ Support function to create coordinates of sources / recievers """
        dist_x, dist_y = (dist_lines, dist) if lines_axis==0 else (dist, dist_lines)
        x, y = np.mgrid[start:stop+dist_x:dist_x, start:stop+dist_y:dist_y,]
        return np.vstack([x.ravel(), y.ravel()]).T

    def calc_active_recievers_mask(sx, sy, reciever_coords, activation_dist):
        """ Support function to get mask of recievers activated by source in position `sx, sy` """
        return np.logical_and(*(np.abs(reciever_coords - (sx, sy)) <= activation_dist).T)

    # Create points for sources and recievers
    source_coords = generate_coordinates(dist_sources//2, sources_size, dist_sources, dist_source_lines, 1)
    reciever_coords = generate_coordinates(-2*activation_dist, sources_size+2*activation_dist,
                                           dist_recievers, dist_reciever_lines)

    # Create and fill up segy spec
    spec = segyio.spec()
    spec.format = kwargs.get('format', 5)
    spec.samples = np.arange(samples)

    # Unstructured mode of SEGY requires to specify number of traces in file - tracecount,
    # which is calculated as number of sources multiplied by number of active reciervers per source
    # and multiplied by a factor of two for a safe margin since it does not affect segy's size on disk
    sx, sy = source_coords[0]
    num_active_recievers = np.sum(calc_active_recievers_mask(sx, sy, reciever_coords, activation_dist))
    tracecount = source_coords.shape[0] * num_active_recievers
    spec.tracecount = tracecount

    # Parse headers' kwargs
    sample_rate = int(kwargs.get('sample_rate', 2000))
    delay = int(kwargs.get('delay', 0))

    with segyio.create(path_segy, spec) as dst_file:
        # Loop over the array and put all the data into new segy file
        TRACE_SEQUENCE_FILE = 0

        with tqdm(total=source_coords.shape[0]) as prog_bar:
            for FieldRecord, (sx, sy) in enumerate(source_coords):
                mask = calc_active_recievers_mask(sx, sy, reciever_coords, activation_dist)
                active_recievers = reciever_coords[mask, :]

                # TODO: maybe add trace with zero offset
                for TraceNumber, (rx, ry) in enumerate(active_recievers):
                    TRACE_SEQUENCE_FILE += 1
                    # Create header
                    header = dst_file.header[TRACE_SEQUENCE_FILE-1]

                    # Fill header
                    header[segyio.TraceField.FieldRecord] = FieldRecord
                    header[segyio.TraceField.TraceNumber] = TraceNumber
                    header[segyio.TraceField.SourceX] = sx
                    header[segyio.TraceField.SourceY] = sy
                    header[segyio.TraceField.GroupX] = rx
                    header[segyio.TraceField.GroupY] = ry

                    offset = int(((sx-rx)**2 + (sy-ry)**2)**0.5) # `int` is faster
                    header[segyio.TraceField.offset] = offset

                    CDP_X = int(sx + rx / 2)
                    CDP_Y = int(sy + ry / 2)
                    header[segyio.TraceField.CDP_X] = CDP_X
                    header[segyio.TraceField.CDP_Y] = CDP_Y

                    INLINE_3D = CDP_X // bin_size
                    CROSSLINE_3D = CDP_Y // bin_size
                    header[segyio.TraceField.INLINE_3D] = INLINE_3D
                    header[segyio.TraceField.CROSSLINE_3D] = CROSSLINE_3D

                    # Fill depth-related fields in header
                    header[segyio.TraceField.TRACE_SAMPLE_COUNT] = samples
                    header[segyio.TraceField.TRACE_SAMPLE_INTERVAL] = sample_rate
                    header[segyio.TraceField.DelayRecordingTime] = delay

                    # Generate trace and write it to file
                    trace = trace_gen(FieldRecord=FieldRecord, TraceNumber=TraceNumber, SourceX=sx, SourceY=sy,
                                           GroupX=rx, GroupY=ry, offset=offset, CDP_X=CDP_X, CDP_Y=CDP_Y,
                                           INLINE_3D=INLINE_3D, CROSSLINE_3D=CROSSLINE_3D, samples=samples,
                                           sample_rate=sample_rate)
                    dst_file.trace[TRACE_SEQUENCE_FILE-1] = trace
                    prog_bar.update(1)

        dst_file.bin = {segyio.BinField.Traces: TRACE_SEQUENCE_FILE,
                        segyio.BinField.Samples: samples,
                        segyio.BinField.Interval: sample_rate}
