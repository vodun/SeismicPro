from functools import partial

from .interactive_map import ScatterMapPlot, BinarizedMapPlot
from ..utils import to_list


class Metric:
    name = None
    min_value = None
    max_value = None
    is_lower_better = None

    @staticmethod
    def calc(*args, **kwargs):
        _ = args, kwargs
        raise NotImplementedError

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class PlottableMetric(Metric):
    vmin = None
    vmax = None
    views = "plot_on_click"
    interactive_scatter_map_class = ScatterMapPlot
    interactive_binarized_map_class = BinarizedMapPlot

    def plot_on_click(self, coords, ax, **kwargs):
        _ = coords, ax, kwargs
        raise NotImplementedError

    def get_views(self, *args, **kwargs):
        _ = args, kwargs
        return [getattr(self, view) for view in to_list(self.views)]


class PartialMetric:
    def __init__(self, metric, *args, **kwargs):
        if not(isinstance(metric, PartialMetric) or isinstance(metric, type) and issubclass(metric, Metric)):
            raise ValueError("metric must be either an instance of PartialMetric or a subclass of Metric")
        self.metric = partial(metric, *args, **kwargs)

    def __getattr__(self, name):
        if name in self.metric.keywords:
            return self.metric.keywords[name]
        return getattr(self.metric.func, name)

    def __call__(self, *args, **kwargs):
        return self.metric(*args, **kwargs)


def define_metric(cls_name="MetricPlaceholder", base_cls=Metric, **kwargs):
    return type(cls_name, (base_cls,), kwargs)
