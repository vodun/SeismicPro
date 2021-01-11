"""File contains implemented abstarct classes for Survay and Gather classes. """
from abs import ABCMeta, abstractmethod

class AbstractSurvay(metaclass=ABCMeta):
    """ Abstract class to check that all nesessery methods are implemented in `Survay` class """
    @abstractmethod
    def get_gather(self):
        pass

    @abstractmethod
    def dump(self):
        pass

    @abstractmethod
    def merge(self):
        pass

    @abstractmethod
    def concat(self):
        pass

    @abstractmethod
    def find_sdc_params(self):
        pass

    @abstractmethod
    def find_equalization_params(self):
        pass


class AbstarctGather(metaclass=ABCMeta):
    """ Abstract class to check that all nesessery methods are implemented in `Gather` class """
    @abstractmethod
    def __getattr__(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def dump(self):
        pass
    @abstractmethod
    def sort(self):
        pass

    @abstractmethod
    def equalize(self):
        pass

    @abstractmethod
    def band_pass_filter(self):
        pass

    @abstractmethod
    def correct_spherical_divergence(self):
        pass

    @abstractmethod
    def drop_zero_traces(self):
        pass

    @abstractmethod
    def hodograph_straightening(self):
        pass

    @abstractmethod
    def mcm(self):
        pass

    @abstractmethod
    def pad_traces(self):
        pass

    @abstractmethod
    def slice_traces(self):
        pass

    @abstractmethod
    def plot(self): # see `seismic_batch.seismic_plot`
        pass

    @abstractmethod
    def plot_gain(self):
        pass

    @abstractmethod
    def plot_spectrum(self):
        pass

    @abstractmethod
    def plot_stats(self): # see `seismic_batch.statistics_plot`
        pass
