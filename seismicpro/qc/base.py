"""Class for metics"""

class BaseQC:
    """Base class for metrics calculation and plotting. """

    def plot(self, *args, **kwargs):
        """ Plot calculated metrics. """
        raise NotImplementedError
