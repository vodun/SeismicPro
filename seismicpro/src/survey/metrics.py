from ..metrics import PlottableMetric


class SurveyAttribute(PlottableMetric):
    def __init__(self, survey, coords_cols, **kwargs):
        super().__init__(**kwargs)
        self.survey = survey.reindex(coords_cols)

    def plot_on_click(self, coords, ax, sort_by=None, **kwargs):
        gather = self.survey.get_gather(coords)
        if sort_by is not None:
            gather = gather.sort(by=sort_by)
        gather.plot(ax=ax, **kwargs)
