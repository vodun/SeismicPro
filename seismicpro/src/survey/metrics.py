from functools import partial

from ..metrics import Metric


class SurveyAttribute(Metric):
    def __init__(self, survey, coords_cols, **kwargs):
        super().__init__(**kwargs)
        self.survey = survey.reindex(coords_cols)

    def plot(self, coords, ax, sort_by=None, **kwargs):
        gather = self.survey.get_gather(coords)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        gather.plot(ax=ax, **kwargs)

    def get_views(self, sort_by=None, **kwargs):
        return [partial(self.plot, sort_by=sort_by)], kwargs
