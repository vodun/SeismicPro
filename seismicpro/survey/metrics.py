"""Implements survey metrics"""

from functools import partial

from ..metrics import Metric


class SurveyAttribute(Metric):
    """A utility metric class that reindexes given survey by `coords_cols` and allows for plotting gathers by their
    coordinates. Does not implement any calculation logic."""
    def __init__(self, name=None):
        super().__init__(name=name)

        # Attributes set after context binding
        self.survey = None

    def bind(self, metric_map, survey):
        self.survey = survey.reindex(metric_map.coords_cols)

    def plot(self, ax, coords, index, sort_by=None, **kwargs):
        """Plot a gather by given `coords`. Optionally sort it."""
        _ = index
        gather = self.survey.get_gather(coords)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        gather.plot(ax=ax, **kwargs)

    def get_views(self, sort_by=None, **kwargs):
        """Return a single view, that plots a gather sorted by `sort_by` by click coordinates."""
        return [partial(self.plot, sort_by=sort_by)], kwargs
