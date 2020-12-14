""" Helper classes for velocity picking QC """

import re

import numpy as np

class VelocityLaw:
    """ Single velocity law """

    def __init__(self, times, velocities):
        times = np.asarray(times)
        velocities = np.asarray(velocities)

        if times.shape != velocities.shape or times.ndim != 1:
            raise AttributeError("Inconsistent dimensions! times.shape is", times.shape,
                                 "velocities.shape is", velocities.shape)

        if np.any(times[1:] - times[:-1] <= 0):
            raise AttributeError("times not increasing!")

        self._times = times
        self._vels = velocities

    @property
    def times(self):
        return self._times

    @property
    def velocities(self):
        return self._vels

    def get_when_increasing(self):
        """ returns times when velocity increases, if any """
        inc_indices = ((self._vels[1:] - self._vels[:-1]) <= 0).nonzero()[0] + 1
        return self._times[inc_indices]

    def prev_time(self, t_curr):
        """ time point in velocity law before given """
        idx = (self._times < t_curr).nonzero()[0]
        if idx.size:
            return self._times[idx[-1]]
        return None

    def next_time(self, t_curr):
        """ time point in velocity law after given """
        idx = (self._times > t_curr).nonzero()[0]
        if idx.size:
            return self._times[idx[0]]
        return None

    def __getitem__(self, time):
        """ returns velocity at given time """
        res = self._vels[self._times == time]
        return res[0] if res.size > 0 else None


class LateralGrid:
    """
    Describes 2D coordinates grid, on which velocity laws are specified
    Lateral grid can be represented in two coordinate sistems:
    1) original,
    2) simplified with origin in the point with smallest X and Y original coordinates
    and unit distance along each coordinate axis being equal to minimum distance between traces on that axis

    lateral coordinates are assumed to be integers
    """

    def __init__(self, coords_list):
        x_list, y_list = list(zip(*coords_list))

        unique_x = np.unique(x_list)
        unique_y = np.unique(y_list)

        self._origin = np.array([unique_x[0], unique_y[0]])

        d_x = np.min(unique_x[1:] - unique_x[:-1])
        d_y = np.min(unique_y[1:] - unique_y[:-1])
        self._delta = np.array([d_x, d_y])

        self._rec_shape = ((np.array([unique_x[-1], unique_y[-1]]) - self._origin)/self._delta).astype(int) + 1

    @property
    def grid_shape(self):
        """
        shape (in the simplified grid) of a minimal rectangle
        that contains all coordinates that initialized LateralGrid
        """
        return self._rec_shape

    def to_grid_coords(self, lat_coords):
        return ((np.array(lat_coords) - self._origin) / self._delta).astype(int)

    def from_grid_coords(self, coords):
        return np.array(coords) * self._delta + self._origin

    def get_window(self, window_center, window_radius):
        """ Square window of given size and center

        Parameters
        ----------
        window_center : tuple of ints
            lateral coordinates of the window's center
        window_radius : int
            number of points on the simplified grid covered by the window in all directions,
            `window_radius == 2` produces 5x5 window.

        Returns
        -------
        list of tuples
            lateral coordnates of traces in the window,
            irrespective of their presence in coordinates list that initialized the LateralGrid
        """
        return [tuple(self.from_grid_coords(self.to_grid_coords(window_center) + np.array([x, y])))
                for x in range(-window_radius, window_radius + 1) for y in range(-window_radius, window_radius + 1)]


class VelocityCube:
    """ Models an area with velocity laws """

    def __init__(self):
        self._velocity_laws = {}
        self._lateral_grid = None

    @property
    def grid_shape(self):
        return self._lateral_grid.grid_shape

    def to_grid_coords(self, lat_coords):
        return self._lateral_grid.to_grid_coords(lat_coords)

    def from_grid_coords(self, coords):
        return self._lateral_grid.from_grid_coords(coords)

    def gen_filled_coordinates(self):
        """ returns generator of coordinates where velocity laws are specified """
        return self._velocity_laws.keys()

    def load(self, fname):
        """ Load velocities from file """
        with open(fname) as f:
            self.parse_seq(f)

    def parse_seq(self, seq):
        """ parse sequense with VFUNC """
        curr_coords = None
        curr_items = []

        def dump(curr_coords, curr_items):
            if curr_coords is not None:
                if curr_coords in self._velocity_laws:
                    raise ValueError(f"Duplicate coordinates {curr_coords}!")

                if len(curr_items) % 2 != 0:
                    raise ValueError("Incorrect format: velocity law consists of pairs of time-speed values, " +
                                     f"thus should have even length. Velocity law at {curr_coords} has odd length!")

                curr_items = np.array(curr_items)

                times = curr_items[0::2]
                velocities = curr_items[1::2]

                self._velocity_laws[curr_coords] = VelocityLaw(times, velocities)

        for line in seq:
            line = line.rstrip(' \r\n')

            if line.startswith('VFUNC'):

                dump(curr_coords, curr_items)

                _, x, y = re.split(r'\W+', line)
                curr_coords = int(x), int(y)
                curr_items = []
            else:
                if not curr_coords:
                    raise ValueError("Wrong format! Coordinates not set")
                curr_items.extend(int(el) for el in re.split(r'\W+', line) if el)

        dump(curr_coords, curr_items)

        self._lateral_grid = LateralGrid(self.gen_filled_coordinates())


    def get_lateral_dispersions(self, window_radius):
        """ velocity-laws-likeliness-related indicators

        Parameters
        ----------
        window_radius : int
            radius of lateral window. See :meth: ~.LateralGrid.get_window

        Returns
        -------
        list of tuples
            for each trace returns a tuple, that consists of its coordinates and indicator values
            for a window with a center in that trace.
            Current indicator values are:
            - Maximum standandard deviation of velocities values in the window, for all times
            - Center velocity law vs mean velocity law for all window Maximum average percentage error, for all times
        """
        res_disp = []
        res_center = []
        coords = list(self.gen_filled_coordinates())

        for x in coords:
            win = self._lateral_grid.get_window(x, window_radius)
            win_laws = {c: self._velocity_laws[c] for c in win if c in self._velocity_laws}

            new_times = np.unique(np.hstack([vl.times for vl in win_laws.values()]))
            new_vels = {c: np.interp(new_times, vl.times, vl.velocities) for c, vl in win_laws.items()}
            center_vels = new_vels[x]
            new_vels = np.vstack(list(new_vels.values()))

            res_center.append(np.abs((np.mean(new_vels, axis=0) - center_vels) / center_vels).max())
            res_disp.append(np.std(new_vels, axis=0).max())

        return coords, res_disp, res_center

    def get_vel_increase_errors(self):
        """ Provides info about velocity laws that have increasing velocity

        Returns
        -------
        list of tuples
            for each occurence of increasing velocity result contains a tuple of
            ('CoordX', 'CoordY', 'Time_prev', 'Vel_prev', 'Time_curr', 'Vel_curr')
        """
        vel_errors = []
        for coords in self.gen_filled_coordinates():
            vl = self._velocity_laws.get(coords)
            for t in vl.get_when_increasing():
                t_prev = vl.prev_time(t)
                vel_errors.append((*coords, t_prev, vl[t_prev], t, vl[t]))

        return vel_errors
