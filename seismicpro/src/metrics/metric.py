class Metric:
    is_lower_better = True
    vmin = None
    vmax = None

    @staticmethod
    def calc(*args, **kwrgs):
        raise NotImplementedError

    @staticmethod
    def plot(*args, ax, **kwrgs):
        raise NotImplementedError

    def coords_to_args(self, x, y):
        raise NotImplementedError
