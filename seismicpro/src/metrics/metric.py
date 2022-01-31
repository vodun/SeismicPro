class Metric:
    name = None
    is_lower_better = True
    vmin = None
    vmax = None

    @staticmethod
    def calc(*args, **kwargs):
        raise NotImplementedError


class PlottableMetric(Metric):
    @staticmethod
    def plot(self, *args, ax, **kwargs):
        raise NotImplementedError

    def coords_to_args(self, coords):
        raise NotImplementedError

    def plot_on_click(self, coords, ax, **kwargs):
        self.plot(*self.coords_to_args(coords), ax=ax, **kwargs)


def define_metric(cls_name="MetricPlaceholder", **kwargs):
    return type(cls_name, (Metric,), kwargs)
