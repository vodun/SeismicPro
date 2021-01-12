from ..batchflow import DatasetIndex


class SeismicIndex(DatasetIndex):
    def __init__(self, *args, **kwargs):
        self.surveys_dict = None
        super().__init__(*args, **kwargs)

    def build_index(self, index=None, surveys=None, *args, **kwargs):
        if index is not None:
            _index = self.build_from_index(index, *args, **kwargs)
        else:
            _index = self.build_from_surveys(surveys, *args, **kwargs)
        return _index

    def build_from_index(self, index, survey_dict):
        self.surveys_dict = survey_dict
        return index

    def build_from_surveys(self, surveys, *args, **kwargs):
        if not isinstance(surveys, (list, tuple)):
            surveys = [surveys]
        self.surveys_dict = {survey.name: survey for survey in surveys}
        index = self.get_unique_index(surveys, *args, **kwargs)
        return index

    @staticmethod
    def get_unique_index(surveys, *args, **kwargs):
        # TODO: write merge
        return surveys[0].headers.index.unique()

    def create_subset(self, index):
        return type(self)(index=index, survey_dict=self.surveys_dict)

    def get_gather(self, survey_name, index):
        if survey_name not in self.surveys_dict:
            err_msg = "Unknown survey name {}, the index contains only {}"
            raise KeyError(err_msg.format(survey_name, ",".join(self.surveys_dict.keys())))
        return self.surveys_dict[survey_name].get_gather(index=index)
