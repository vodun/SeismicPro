"""General survey processing utils"""

import numpy as np
from numba import njit, prange


@njit(nogil=True)
def calculate_trace_stats(trace):
    """Calculate min, max, mean and var of trace amplitudes."""
    trace_min = trace_max = trace[0]

    # Traces are generally centered around zero so variance is calculated in a single pass by accumulating sum and
    # sum of squares of trace amplitudes as float64 for numerical stability
    trace_sum = np.float64(trace[0])
    trace_sum_sq = trace_sum**2

    for sample in trace[1:]:
        trace_min = min(sample, trace_min)
        trace_max = max(sample, trace_max)
        sample64 = np.float64(sample)
        trace_sum += sample64
        trace_sum_sq += sample64**2
    trace_mean = trace_sum / len(trace)
    trace_var = trace_sum_sq / len(trace) - trace_mean**2
    return trace_min, trace_max, trace_mean, trace_var


@njit(nogil=True, parallel=True)
def ibm_to_ieee(hh, hl, lh, ll):
    """Convert 4 arrays representing individual bytes of IBM 4-byte floats into a single array of floats. Input arrays
    are ordered from most to least significant bytes and have `np.uint8` dtypes. The result is returned as an
    `np.float32` array."""
    res = np.empty_like(hh, dtype=np.float32)
    for i in prange(res.shape[0]):  # pylint: disable=not-an-iterable
        for j in prange(res.shape[1]):  # pylint: disable=not-an-iterable
            mant = (((np.int32(hl[i, j]) << 8) | lh[i, j]) << 8) | ll[i, j]
            if hh[i, j] & 0x80:
                mant = -mant
            exp16 = (np.int8(hh[i, j]) & np.int8(0x7f)) - 70
            res[i, j] = mant * 16.0**exp16
    return res

def validate_trace_headers(headers, full=False):
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

    if len({"FieldRecord", "TraceNumber"} & avalible_columns) == 2:
        n_unique_ids = (~headers[["FieldRecord", "TraceNumber"]].duplicated()).sum()
        has_unique_trace_id = n_unique_ids == n_traces
        if not has_unique_trace_id:
            msg_list.append("Non-unique traces indentifier (FieldRerocd, TraceNumber)")

    if len({"FieldRecord", *shot_cols} & avalible_columns) == 3:
        fr_with_coords = headers[["FieldRecord", *shot_cols]].drop_duplicates()
        n_unique_fr = fr_with_coords["FieldRecord"].nunique()
        has_unique_coords = len(fr_with_coords) == n_unique_fr
        if not has_unique_coords:
            msg_list.append("Several pairs of coordinates (SourceX, SourceY) for single FieldRecond")

    # Check that Eqlidian distance calculated based on the coords from shot to receiver is close to the one stored
    # in trace headers
    if len({*shot_cols, *rec_cols, "offset"} & avalible_columns) == 5:
        calculated_offsets = np.sqrt(np.sum((headers[shot_cols].values - headers[rec_cols].values)**2, axis=1))
        # Avoiding small offsets since they may leads to false positive estimation.
        mask = headers["offset"] > 20
        real_offsets = headers["offset"].values[mask]
        print(calculated_offsets, real_offsets)
        has_correct_offsets = np.all(np.abs(calculated_offsets[mask] - real_offsets) / real_offsets < 0.1)
        if not has_correct_offsets:
            msg_list.append("Mismatch of the distance between shots and receivers posinitons for each trace"
                            " and offsets in headers")

    if len({*cdp_cols, *bin_cols} & avalible_columns) == 4:
        unique_inline_cdp = headers[[*bin_cols, *cdp_cols]].drop_duplicates()
        unique_cdp = unique_inline_cdp[cdp_cols].drop_duplicates()
        has_unique_inline_to_cdp = len(unique_inline_cdp) == len(unique_cdp)
        if not has_unique_inline_to_cdp:
            msg_list.append("Non-unique mapping of geographic (CDP) to binary (INLINE/CROSSLINE) coordinates")

    if len({*shot_cols, *rec_cols, *cdp_cols} & avalible_columns) == 6:
        raw_cdp = (headers[shot_cols].values + headers[rec_cols].values) / 2
        has_consistent_geo_coords = np.all((raw_cdp - headers[cdp_cols].values) / headers[cdp_cols].values < 0.1)
        if not has_consistent_geo_coords:
            msg_list.append("Unconsistent range of some geographic coordinates"
                            " (SourceX, SourceY, GroupX, GroupY, CDP_X, CDP_Y)")

    if len({*shot_cols, "SourceSurfaceElevation"} & avalible_columns) == 3:
        ushot_elev_coords = headers[[*shot_cols, "SourceSurfaceElevation"]].drop_duplicates()
        ushot_coords = ushot_elev_coords[shot_cols].drop_duplicates()
        has_shot_uniq_elevs = len(ushot_elev_coords) == len(ushot_coords)
        if not has_shot_uniq_elevs:
            msg_list.append("Different surface elevation (SourceSurfaceElevation) for at least one shot")

    if len({*rec_cols, "ReceiverGroupElevation"} & avalible_columns) == 3:
        urec_elev_coords = headers[[*rec_cols, "ReceiverGroupElevation"]].drop_duplicates()
        urec_coords = urec_elev_coords[rec_cols].drop_duplicates()
        has_rec_uniq_elevs = len(urec_elev_coords) == len(urec_coords)
        if not has_rec_uniq_elevs:
            msg_list.append("Different surface elevation (ReceiverGroupElevation) for at least one receiver")

    if len({*shot_cols, *rec_cols, "ReceiverGroupElevation", "SourceSurfaceElevation"} & avalible_columns) == 6:
        shot_elevations = ushot_elev_coords.values[:, 2]
        shot_interp = IDWInterpolator(ushot_elev_coords.values[:, :2], shot_elevations, neighbors=3)
        mask = shot_elevations > 0
        rec_by_shot = np.abs(shot_interp(ushot_elev_coords.values[:, :2][mask]) - shot_elevations[mask])
        has_correct_elevations = np.all(rec_by_shot / shot_elevations[mask] < 0.1)
        if not has_correct_elevations:
            msg_list.append("Unconsistent values in elevations-related headers"
                            "(ReceiverGroupElevation, SourceSurfaceElevation)")

    if len(msg_list) > 0:
        line = "\n" + "#"*120
        msg = line +  f"\n## {'The loaded Survey has the following problems with trace headers:': <115}##"
        msg = f"{msg}" + "".join([f"\n## {i+1}. {msg:<112}##" for i, msg in zip(range(len(msg_list)), msg_list)])
        msg += line
        warnings.warn(msg, Warning)
