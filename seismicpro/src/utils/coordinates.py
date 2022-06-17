"""Coordinates-related utility functions and classes"""

import numpy as np

from .general_utils import to_list


INDEX_TO_COORDS = {
    # Shot index
    "FieldRecord": ("SourceX", "SourceY"),
    ("SourceX", "SourceY"): ("SourceX", "SourceY"),

    # Receiver index
    ("GroupX", "GroupY"): ("GroupX", "GroupY"),

    # Trace index
    "TRACE_SEQUENCE_FILE": ("CDP_X", "CDP_Y"),
    ("FieldRecord", "TraceNumber"): ("CDP_X", "CDP_Y"),
    ("SourceX", "SourceY", "GroupX", "GroupY"): ("CDP_X", "CDP_Y"),

    # Bin index
    "CDP": ("CDP_X", "CDP_Y"),
    ("CDP_X", "CDP_Y"): ("CDP_X", "CDP_Y"),
    ("INLINE_3D", "CROSSLINE_3D"): ("INLINE_3D", "CROSSLINE_3D"),
    ("SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"): ("SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D"),
}
# Ignore order of elements in each key
INDEX_TO_COORDS = {frozenset(to_list(key)): val for key, val in INDEX_TO_COORDS.items()}


def get_coords_cols(index_cols):
    """Return headers columns to get coordinates from depending on the type of headers index. See the mapping in
    `INDEX_TO_COORDS`."""
    coords_cols = INDEX_TO_COORDS.get(frozenset(to_list(index_cols)))
    if coords_cols is None:
        raise KeyError(f"Unknown coordinates columns for {index_cols} index")
    return coords_cols


GEOGRAPHIC_COORDS = {("SourceX", "SourceY"), ("GroupX", "GroupY"), ("CDP_X", "CDP_Y")}
LINE_COORDS = {("INLINE_3D", "CROSSLINE_3D"), ("SUPERGATHER_INLINE_3D", "SUPERGATHER_CROSSLINE_3D")}
ALLOWED_COORDS = GEOGRAPHIC_COORDS + LINE_COORDS


class Coordinates:
    """Define spatial coordinates of an object."""

    def __init__(self, coords, names):
        coords = tuple(to_list(coords))
        if len(coords) != 2:
            raise ValueError("Exactly two coordinates must be passed.")
        self.coords = coords

        names = tuple(to_list(names))
        if len(names) != 2:
            raise ValueError("Exactly two names must be passed.")
        if names not in ALLOWED_COORDS:
            raise ValueError(f"Unknown coordinates names {names}.")
        self.names = names

        self.is_geographic = names in GEOGRAPHIC_COORDS

    def __repr__(self):
        return f"Coordinates({self.coords}, names={self.names})"

    def __str__(self):
        return f"({self.names[0]}: {self.coords[0]}, {self.names[1]}: {self.coords[1]})"

    def __iter__(self):
        return iter(self.coords)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, key):
        return self.coords[key]

    def __array__(self, dtype=None):
        return np.array(self.coords, dtype=dtype)
