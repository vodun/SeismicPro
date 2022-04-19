"""Custom headers optionally added by SeismicPro to those from SEG-Y file"""
from .stacking_velocity import StackingVelocity


HDR_DEAD_TRACE = 'DeadTrace'
HDR_FIRST_BREAK = 'FirstBreak'

DEFAULT_TIMES = [0.0, 100.0, 700.0, 1000.0, 1400.0, 1800.0, 1950.0, 4200.0, 7000.0]
DEFAULT_VELOCITIES = [1524.0, 1524.0, 1924.5, 2184.0, 2339.6, 2676.0, 2889.5, 3566.0, 4785.3]
DEFAULT_STACKING_VELOCITY  = StackingVelocity.from_points(times=DEFAULT_TIMES, velocities=DEFAULT_VELOCITIES)
