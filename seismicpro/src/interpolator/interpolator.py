from scipy.interpolate import Rbf

class RBFInterpolator:
    def __init__(self, x, y, **kwargs):
        self.rbf = Rbf(x[:, 0], x[:, 1], y, **kwargs)

    def interpolate(self, x):
        return self.rbf(x[:, 0], x[:, 1])
