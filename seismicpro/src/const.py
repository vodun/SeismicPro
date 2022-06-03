"""Package-level constants"""

from .stacking_velocity import StackingVelocity


EPS = 1e-10

# Size of trace headers in a SEG-Y file in bytes
TRACE_HEADER_SIZE = 240


# Valid aliases for file endianness and corresponding format string prefixes to be passed to struct.unpack
ENDIANNESS = {
    "big": ">",
    "msb": ">",
    "little": "<",
    "lsb": "<",
}


# Custom headers optionally added by SeismicPro to those from SEG-Y file
HDR_DEAD_TRACE = 'DeadTrace'
HDR_FIRST_BREAK = 'FirstBreak'


# Default velocity for spherical divergence correction as provided in Paradigm.
# times are measured in milliseconds, velocities in meters / second.
DEFAULT_VELOCITY = StackingVelocity.from_points(
                                times=[0.0, 100.0, 700.0, 1000.0, 1400.0, 1800.0, 1950.0, 4200.0, 7000.0],
                                velocities=[1524.0, 1524.0, 1924.5, 2184.0, 2339.6, 2676.0, 2889.5, 3566.0, 4785.3])
