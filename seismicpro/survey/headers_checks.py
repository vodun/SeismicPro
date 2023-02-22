"""Contains methods to validate trace headers for consistency"""

import warnings
from textwrap import wrap

import numpy as np
from sklearn.neighbors import RadiusNeighborsRegressor

from ..utils import to_list


def warn_list(title, warning_list, width=80):
    """Warn about several issues listed in `warning_list`."""
    n_warnings = len(warning_list)
    if n_warnings == 0:
        return
    ix_len = len(str(n_warnings))
    wrap_space = 3 + ix_len
    wrap_sep = "\n" + " " * wrap_space
    line_sep = "\n\n" + "-" * width
    warning_list = [wrap_sep.join(wrap(warn_str, width=width-wrap_space)) for warn_str in warning_list]
    warning_msg = line_sep + "\n\n" + "\n".join(wrap(title, width=width))
    warning_msg += "".join([f"\n\n {i+1:{ix_len}d}. {warn_str}" for i, warn_str in enumerate(warning_list)]) + line_sep
    warnings.warn(warning_msg)


# pylint: disable-next=too-many-statements
def validate_trace_headers(headers, offset_atol=10, cdp_atol=10, elevation_atol=5, elevation_radius=50):
    """Validate trace headers for consistency. `headers` `DataFrame` is expected to have reset index."""
    n_traces = len(headers)
    if n_traces == 0:
        return

    msg_list = []

    shot_coords_cols = ["SourceX", "SourceY"]
    rec_coords_cols = ["GroupX", "GroupY"]
    cdp_coords_cols = ["CDP_X", "CDP_Y"]
    bin_coords_cols = ["INLINE_3D", "CROSSLINE_3D"]

    loaded_columns = set(headers.columns)
    non_empty_columns = set(headers.columns[headers.any(axis=0)])
    empty_columns = loaded_columns - non_empty_columns
    if empty_columns:
        msg_list.append("Empty headers: " + ", ".join(empty_columns))

    if {"FieldRecord", "TraceNumber"} <= non_empty_columns:
        n_duplicated = headers.duplicated(["FieldRecord", "TraceNumber"], keep=False).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique traces identifier (FieldRecord, TraceNumber) for {n_duplicated} traces "
                            f"({(n_duplicated / n_traces):.02f}%)")

    if "SourceUpholeTime" in non_empty_columns:
        n_neg_uphole_time = (headers["SourceUpholeTime"] < 0).sum()
        if n_neg_uphole_time:
            msg_list.append(f"Negative uphole times for {n_neg_uphole_time} traces "
                            f"({(n_neg_uphole_time / n_traces):.02f}%)")

    if "SourceDepth" in non_empty_columns:
        n_neg_uphole_depth = (headers["SourceUpholeTime"] < 0).sum()
        if n_neg_uphole_depth:
            msg_list.append(f"Negative uphole depths for {n_neg_uphole_depth} traces "
                            f"({(n_neg_uphole_depth / n_traces):.02f}%)")

    if {"SourceUpholeTime", "SourceDepth"} <= loaded_columns:
        zero_time_mask = np.isclose(headers["SourceUpholeTime"], 0)
        zero_depth_mask = np.isclose(headers["SourceDepth"], 0)
        n_zero_time = (zero_time_mask[~zero_depth_mask]).sum()
        n_zero_depth = (zero_depth_mask[~zero_time_mask]).sum()
        if n_zero_time:
            msg_list.append(f"Zero uphole time for non-zero uphole depth for {n_zero_time} traces "
                            f"({(n_zero_time / n_traces):.02f}%)")
        if n_zero_depth:
            msg_list.append(f"Zero uphole depth for non-zero uphole time for {n_zero_depth} traces "
                            f"({(n_zero_depth / n_traces):.02f}%)")

    if "offset" in non_empty_columns:
        n_neg_offsets = (headers["offset"] < 0).sum()
        if n_neg_offsets:
            msg_list.append(f"Negative offsets for {n_neg_offsets} traces ({(n_neg_offsets / n_traces):.02f}%)")

    if {*shot_coords_cols, *rec_coords_cols, "offset"} <= non_empty_columns:
        shot_coords = headers[shot_coords_cols].to_numpy()
        rec_coords = headers[rec_coords_cols].to_numpy()
        calculated_offsets = np.sqrt(np.sum((shot_coords - rec_coords)**2, axis=1))
        close_mask = np.isclose(calculated_offsets, headers["offset"].abs(), rtol=0, atol=offset_atol)
        n_diff = (~close_mask).sum()
        if n_diff:
            msg_list.append("Distance between source (SourceX, SourceY) and receiver (GroupX, GroupY) differs from "
                            f"the corresponding offset by more than {offset_atol} meters for {n_diff} traces "
                            f"({(n_diff / n_traces):.02f}%)")

    if {*shot_coords_cols, *rec_coords_cols, *cdp_coords_cols} <= non_empty_columns:
        calculated_cdp = (headers[shot_coords_cols].to_numpy() + headers[rec_coords_cols].to_numpy()) / 2
        close_mask = np.sqrt(np.sum((calculated_cdp - headers[cdp_coords_cols])**2, axis=1)) <= cdp_atol
        n_diff = (~close_mask).sum()
        if n_diff:
            msg_list.append("A midpoint between source (SourceX, SourceY) and receiver (GroupX, GroupY) differs from "
                            f"the corresponding coordinates (CDP_X, CDP_Y) by more than {cdp_atol} meters for "
                            f"{n_diff} traces ({(n_diff / n_traces):.02f}%)")

    if {*shot_coords_cols, "SourceSurfaceElevation"} <= non_empty_columns:
        unique_shot_elevations = headers[shot_coords_cols + ["SourceSurfaceElevation"]].drop_duplicates()
        n_uniques = unique_shot_elevations.groupby(shot_coords_cols).size()
        n_duplicated = (n_uniques > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique surface elevation (SourceSurfaceElevation) for {n_duplicated} source "
                            f"locations ({(n_duplicated / len(n_uniques)):.02f}%)")

    if {*rec_coords_cols, "ReceiverGroupElevation"} <= non_empty_columns:
        unique_rec_elevations = headers[rec_coords_cols + ["ReceiverGroupElevation"]].drop_duplicates()
        n_uniques = unique_rec_elevations.groupby(rec_coords_cols).size()
        n_duplicated = (n_uniques > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique surface elevation (ReceiverGroupElevation) for {n_duplicated} receiver "
                            f"locations ({(n_duplicated / len(n_uniques)):.02f}%)")

    if {*shot_coords_cols, *rec_coords_cols, "ReceiverGroupElevation", "SourceSurfaceElevation"} <= non_empty_columns:
        elevations = np.concatenate([unique_shot_elevations.to_numpy(), unique_rec_elevations.to_numpy()])
        rnr = RadiusNeighborsRegressor(radius=elevation_radius).fit(elevations[:, :2], elevations[:, 2])
        close_mask = np.isclose(rnr.predict(elevations[:, :2]), elevations[:, 2], rtol=0, atol=elevation_atol)
        n_diff = (~close_mask).sum()
        if n_diff:
            msg_list.append("Surface elevations of sources (SourceSurfaceElevation) and receivers "
                            f"(ReceiverGroupElevation) differ by more than {elevation_atol} meters within spatial "
                            f"radius of {elevation_radius} meters for {n_diff} sensor locations "
                            f"({(n_diff / len(elevations)):.02f}%)")

    if {*cdp_coords_cols, *bin_coords_cols} <= non_empty_columns:
        unique_cdp_bin = headers[cdp_coords_cols + bin_coords_cols].drop_duplicates()
        n_cdp_per_bin = unique_cdp_bin.groupby(bin_coords_cols).size()
        n_duplicated = (n_cdp_per_bin > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique midpoint coordinates (CDP_X, CDP_Y) for {n_duplicated} bins "
                            f"({(n_duplicated / len(n_cdp_per_bin)):.02f}%)")
        n_bin_per_cdp = unique_cdp_bin.groupby(cdp_coords_cols).size()
        n_duplicated = (n_bin_per_cdp > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique bin (INLINE_3D, CROSSLINE_3D) for {n_duplicated} midpoint locations "
                            f"({(n_duplicated / len(n_bin_per_cdp)):.02f}%)")

    warn_list("The survey has the following inconsistencies in trace headers:", msg_list)


def validate_source_headers(headers, source_id_cols=None):
    """Validate source-related trace headers for consistency. `headers` `DataFrame` is expected to have reset index."""
    n_traces = len(headers)
    if n_traces == 0:
        return

    if source_id_cols is None:
        return
    source_id_cols = to_list(source_id_cols)

    empty_id_mask = (headers[source_id_cols] == 0).all(axis=0)
    if empty_id_mask.any():
        empty_id_cols = ", ".join(empty_id_mask[empty_id_mask].index)
        warnings.warn(f"No checks are performed since the following source ID headers are empty: {empty_id_cols}")

    msg_list = []
    coords_cols = ["SourceX", "SourceY"]
    loaded_columns = set(headers.columns)

    if set(source_id_cols) != set(coords_cols) and {*source_id_cols, *coords_cols} <= loaded_columns:
        unique_coords = headers[source_id_cols + coords_cols].drop_duplicates()
        n_uniques = unique_coords.groupby(source_id_cols).size()
        n_duplicated = (n_uniques > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique source coordinates (SourceX, SourceY) for {n_duplicated} sources "
                            f"({(n_duplicated / len(n_uniques)):.02f}%)")

    if {*source_id_cols, "SourceSurfaceElevation"} <= loaded_columns:
        unique_elevations = headers[source_id_cols + ["SourceSurfaceElevation"]].drop_duplicates()
        n_uniques = unique_elevations.groupby(source_id_cols).size()
        n_duplicated = (n_uniques > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique surface elevation (SourceSurfaceElevation) for {n_duplicated} sources "
                            f"({(n_duplicated / len(n_uniques)):.02f}%)")

    if {*source_id_cols, "SourceUpholeTime"} <= loaded_columns:
        unique_uphole_time = headers[source_id_cols + ["SourceUpholeTime"]].drop_duplicates()
        n_uniques = unique_uphole_time.groupby(source_id_cols).size()
        n_duplicated = (n_uniques > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique source uphole time (SourceUpholeTime) for {n_duplicated} sources "
                            f"({(n_duplicated / len(n_uniques)):.02f}%)")

    if {*source_id_cols, "SourceDepth"} <= loaded_columns:
        unique_depth = headers[source_id_cols + ["SourceDepth"]].drop_duplicates()
        n_uniques = unique_depth.groupby(source_id_cols).size()
        n_duplicated = (n_uniques > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique source depth (SourceDepth) for {n_duplicated} sources "
                            f"({(n_duplicated / len(n_uniques)):.02f}%)")

    warn_list("Selected source ID columns result in the following inconsistencies of trace headers:", msg_list)


def validate_receiver_headers(headers, receiver_id_cols=None):
    """Validate receiver-related trace headers for consistency. `headers` `DataFrame` is expected to have reset
    index."""
    n_traces = len(headers)
    if n_traces == 0:
        return

    if receiver_id_cols is None:
        return
    receiver_id_cols = to_list(receiver_id_cols)

    empty_id_mask = (headers[receiver_id_cols] == 0).all(axis=0)
    if empty_id_mask.any():
        empty_id_cols = ", ".join(empty_id_mask[empty_id_mask].index)
        warnings.warn(f"No checks are performed since the following receiver ID headers are empty: {empty_id_cols}")

    msg_list = []
    coords_cols = ["GroupX", "GroupY"]
    loaded_columns = set(headers.columns)

    if set(receiver_id_cols) != set(coords_cols) and {*receiver_id_cols, *coords_cols} <= loaded_columns:
        unique_coords = headers[receiver_id_cols + coords_cols].drop_duplicates()
        n_uniques = unique_coords.groupby(receiver_id_cols).size()
        n_duplicated = (n_uniques > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique receiver coordinates (GroupX, GroupY) for {n_duplicated} receivers "
                            f"({(n_duplicated / len(n_uniques)):.02f}%)")

    if {*receiver_id_cols, "ReceiverGroupElevation"} <= loaded_columns:
        unique_elevations = headers[receiver_id_cols + ["ReceiverGroupElevation"]].drop_duplicates()
        n_uniques = unique_elevations.groupby(receiver_id_cols).size()
        n_duplicated = (n_uniques > 1).sum()
        if n_duplicated:
            msg_list.append(f"Non-unique surface elevation (ReceiverGroupElevation) for {n_duplicated} receivers "
                            f"({(n_duplicated / len(n_uniques)):.02f}%)")

    warn_list("Selected receiver ID columns result in the following inconsistencies of trace headers:", msg_list)
