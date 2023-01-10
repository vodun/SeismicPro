"""Classes for stacking velocity analysis"""

from .stacking_velocity import StackingVelocity
from .stacking_velocity_field import StackingVelocityField
from .velocity_model import calculate_stacking_velocity

# Default velocity for spherical divergence correction as provided in Paradigm.
DEFAULT_SDC_VELOCITY = StackingVelocity(
    times=[0.0, 100.0, 700.0, 1000.0, 1400.0, 1800.0, 1950.0, 4200.0, 7000.0],  # milliseconds
    velocities=[1524.0, 1524.0, 1924.5, 2184.0, 2339.6, 2676.0, 2889.5, 3566.0, 4785.3]  # meters/second
)
