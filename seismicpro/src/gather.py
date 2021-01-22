""" File with gather class. """
import numpy as np
import matplotlib.pyplot as plt

from .abstract_classes import AbstractGather

TRACE_UID_HEADER = 'TRACE_SEQUENCE_FILE'



class Gather(AbstractGather):
    """ !! """
    def __init__(self, headers, path=None, name=None,  data=None):
        self.headers = headers
        self.path = path
        self.name = name
        self.data = data

    def __getattr__(self):
        pass

    def dump(self):
        pass

    def sort(self, by):
        arg = np.argsort(self.headers[by].values, kind='stable')

        self.data = self.data[arg]
        self.headers = self.headers.loc[arg]
        return self

    def equalize(self):
        pass

    def band_pass_filter(self):
        pass

    def correct_spherical_divergence(self):
        pass

    def drop_zero_traces(self):
        pass

    def hodograph_straightening(self):
        pass

    def mcm(self):
        pass

    def pad_traces(self):
        pass

    def slice_traces(self):
        pass

    def plot(self):
        kwargs = {
            'cmap': 'gray',
            'vmin': np.quantile(self.data, 0.1),
            'vmax': np.quantile(self.data, 0.9),
            'aspect': 'auto',
        }
        plt.figure(figsize=(10, 7))
        plt.imshow(self.data.T, **kwargs)


    def plot_gain(self):
        pass

    def plot_spectrum(self):
        pass

    def plot_stats(self):
        pass
