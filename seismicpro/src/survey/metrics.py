from ..metrics import PlottableMetric


class SurveyAttribute(PlottableMetric):
    def __init__(self, survey, coords_cols, sort_by=None, **kwargs):
        super().__init__(**kwargs)
        self.survey = survey.reindex(coords_cols)
        self.sort_by = sort_by

    def plot_on_click(self, coords, ax, **kwargs):
        gather = self.survey.get_gather(coords)
        if self.sort_by is not None:
            gather = gather.sort(by=self.sort_by)
        gather.plot(ax=ax, **kwargs)
