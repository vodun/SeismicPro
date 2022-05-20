from functools import partial

import numpy as np
from scipy import optimize
from scipy.stats import norm

from ..decorators import plotter
from ..utils.plot_utils import add_colorbar
from ..utils import set_ticks, set_text_formatting
from ..const import HDR_FIRST_BREAK


class GeometryControl:
    def __init__(self):
        self.gather = None
        self.weathering_velocity = None
        self.cut_gather = None
        self.direction = None
        self.transverse = None
        self.difference = None
        self.data = None
        self.norm_data = None
        self._model_params = None
        self._debug = None

    @classmethod
    def by_grid_search(cls, gather, n_layers=1, grid_steps=36, **kwargs):
        self = cls()

        self.gather = gather.sort(by='offset')
        self.weathering_velocity = gather.calculate_weathering_velocity(n_layers=n_layers)
        self.gather = self.calculate_lmo_difference(self.gather, wv=self.weathering_velocity)
        self.cut_gather = self.slice_gather(self.gather)
        self.calculate_direction_by_grid_search(self.cut_gather, grid_steps=grid_steps)
        return self

    @classmethod
    def by_minimize(cls, gather, n_layers=1, **kwargs):
        self = cls()

        self._debug = []
        self.gather = gather.sort(by='offset')
        self.weathering_velocity = gather.calculate_weathering_velocity(n_layers=n_layers)
        self.gather = self.calculate_lmo_difference(self.gather, wv=self.weathering_velocity)
        self.cut_gather = self.slice_gather(self.gather)
        partial_loss_func = partial(self.loss_calculation, loss=kwargs.pop('loss', 'L1'))
        self.calculate_direction_by_minimize(self.cut_gather, partial_loss_func, **kwargs)

        return self

    @staticmethod
    def calculate_group_vectors(gather):
        """Calculate vector for source point to group point."""
        dx = gather['GroupX'] - gather['SourceX']
        dy = gather['GroupY'] - gather['SourceY']
        return dx.ravel(), dy.ravel()

    @staticmethod
    def calculate_lmo_difference(gather, wv=None, first_break_col=HDR_FIRST_BREAK):  # maybe move in metrics
        """Calculate difference between first break value and the expected arrival time (weathering velocity curve)."""
        if wv is None:
            wv = gather.calculate_weathering_velocity(n_layers=1, loss='cauchy')
        gather['lmo_diff'] = gather[first_break_col] - wv(gather.offsets).reshape((-1, 1))
        return gather

    @staticmethod
    def slice_gather(gather, n=100):
        """Return gather with N first traces."""
        return gather[:n]

    # worse algo
    def calculate_direction_by_grid_search(self, gather, grid_steps=360, at_least=10, lmo_diff_col="lmo_diff"):
        """Calculate direction of expected true source point by grid search."""
        x = np.linspace(0, 2*np.pi, grid_steps)
        base = np.array([np.cos(x), np.sin(x)]).T
        vectors = np.array(self.calculate_group_vectors(gather)).T
        self.data = np.array([*vectors.T, gather[lmo_diff_col].ravel()]).T
        direction_max = np.zeros(2)
        direction_min = np.zeros(2)
        diff_max = 0
        diff_min = np.inf
        diff_max_std = 0
        diff_min_std = 0
        for i in range(grid_steps):
            signs = np.sign(np.cross(vectors, base[i]))
            if sum(signs == 1) >= at_least and sum(signs == -1) >= at_least:
                gather_1 = gather[signs == 1]
                gather_2 = gather[signs == -1]
                diff_mu, std = self.calculate_columns_diff(gather_1, gather_2)
                if diff_mu > diff_max:
                    direction_max = base[i]
                    diff_max = diff_mu
                    diff_max_std = std
                if diff_mu < diff_min:
                    direction_min = base[i]
                    diff_min = diff_mu
                    diff_min_std = std
        self.transverse = direction_max
        normal_turn = np.array([[0, 1], [-1, 0]])
        self.direction = np.matmul(direction_max, normal_turn)
        self.difference = diff_max

    @staticmethod
    def calculate_columns_diff(gather_1, gather_2, col='lmo_diff'):
        """Calculate mean between same column of two gathers."""
        mu_1, std_1 = norm.fit(gather_1[col])
        mu_2, std_2 = norm.fit(gather_2[col])
        return mu_2 - mu_1, max(std_1, std_2)

    def calculate_direction_by_minimize(self, gather, loss_func, lmo_diff_col="lmo_diff", **kwargs): # algo
        """Calculate direction of expected true source point by `scipy.minimize`."""
        self.data = np.array([*self.calculate_group_vectors(gather), gather[lmo_diff_col].ravel()]).T
        coord_norm = np.max(np.abs(self.data[:, :2]))
        times_norm = np.max(np.abs(self.data[:, 2]))
        self.norm_data = self.data #/ np.array([coord_norm, coord_norm, times_norm])  # normalization should be there
        self._model_params = optimize.minimize(loss_func, x0=[0, 0], method='Nelder-Mead', **kwargs)
        direction = self._model_params.x
        value = np.matmul(direction, direction.T) ** .5  # L2 norm
        normal_turn = np.array([[0, 1], [-1, 0]])

        self.direction = direction / value
        self.transverse = np.matmul(self.direction, normal_turn)
        self.difference = self.get_difference()

    def get_difference(self):
        vectors = np.array(self.calculate_group_vectors(self.cut_gather)).T
        signs = np.sign(np.cross(vectors, self.transverse))
        if sum(signs == 1) >= 1 and sum(signs == -1) >= 1:
            value_1 = self.data[:, 2][signs == 1]
            value_2 = self.data[:, 2][signs == -1]
            diff = value_1.mean() - value_2.mean()
            return np.abs(diff)
        return np.nan

    def loss_calculation(self, args, loss='L1'):  # func : z = a*x + b*y, args : [a, b]
        expected = np.matmul(args, self.norm_data[:, :2].T)
        self._debug.append(expected)
        true = self.data[:, 2]
        diff_abs = np.abs(true - expected)
        if loss =='L1':
            return diff_abs.mean()
        if loss == 'soft_L1':
            return 2 * ((1 + diff_abs) ** .5 - 1).mean()
        if loss == 'cauchy':
            return np.log(diff_abs + 1).mean()
        raise ValueError

    @plotter(figsize=(10, 5))
    def plot(self, *, ax=None, title=None, x_ticker=None, y_ticker=None, res=None, c=None, **kwargs):
        """Plot the receiver points, lmo shift, transverse line and quiver towards the increasing lmo shift."""
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)
        set_ticks(ax, "x", tick_labels=None, label="source_x", **x_ticker)
        set_ticks(ax, "y", tick_labels=None, label="source_y", **y_ticker)

        if c is None:
            c = self.data[:, 2]
        source_x = np.array((self.gather['SourceX'][0][0], self.gather['SourceX'][0][0]))
        source_y = np.array((self.gather['SourceY'][0][0], self.gather['SourceY'][0][0]))
        ax.scatter(self.data[:, 0] + source_x[0], self.data[:, 1] + source_y[0], c=c)
        ax.plot(np.array([-self.transverse[0], self.transverse[0]]) * self.gather.offsets.max() / 2 + source_x,
                np.array([-self.transverse[1], self.transverse[1]]) * self.gather.offsets.max() / 2 + source_y)
        ax.quiver(source_x[0], source_y[0], self.direction[0], self.direction[1],
                  scale=30/self.difference, scale_units='inches')
        # add_colorbar(ax, img, colorbar=True, y_ticker=None)
        ax.set_title(**{"label": None, **title})
