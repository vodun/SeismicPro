from functools import reduce

import pandas as pd

from ..batchflow import DatasetIndex


class SeismicIndex(DatasetIndex):
    def __init__(self, *args, **kwargs):
        self.surveys_dict = None
        self.headers = None
        super().__init__(*args, **kwargs)

    def build_index(self, index=None, surveys=None, *args, **kwargs):
        if index is None and surveys is None:  # Create empty index
            _index = None
        elif index is not None:
            _index = self.build_from_index(index, *args, **kwargs)
        elif not isinstance(surveys, (list, tuple)):
            _index = self.build_from_survey(surveys)
        else:
            _index = self.build_from_surveys(surveys, *args, **kwargs)
        return _index

    def build_from_index(self, index, survey_dict, headers):
        self.surveys_dict = survey_dict
        self.headers = headers.loc[index]
        return index

    def build_from_survey(self, survey):
        self.surveys_dict = {survey.name: [survey]}
        headers = survey.headers.copy()
        headers["CONCAT_ID"] = 0
        headers.set_index("CONCAT_ID", append=True, inplace=True)
        n_index_columns = len(headers.index.names)
        order = [n_index_columns - 1] + list(range(n_index_columns - 1))
        self.headers = headers.reorder_levels(order)
        index = self.headers.index.unique()
        return index

    def build_from_surveys(self, surveys, mode, *args, **kwargs):
        if mode in ("merge", "m"):
            index = SeismicIndex.merge(surveys, *args, **kwargs)
        elif mode in ("concat", "c"):
            index = SeismicIndex.concat(surveys, *args, **kwargs)
        else:
            raise ValueError("Unknown mode {}".format(mode))
        return self.build_from_index(index=index.index, survey_dict=index.surveys_dict, headers=index.headers)

    @staticmethod
    def surveys_to_indices(surveys):
        survey_indices = []
        for survey in surveys:
            if not isinstance(survey, SeismicIndex):
                survey = SeismicIndex(surveys=survey)
            survey_indices.append(survey)
        return survey_indices

    def get_pos(self, index):
        # Only single index!
        return self._pos[index]

    @classmethod
    def merge_two_indices(cls, x, y, *args, **kwargs):
        intersect_keys = x.surveys_dict.keys() & y.surveys_dict.keys()
        if intersect_keys:
            raise ValueError("Only surveys with unique names can be merged,"
                             "but {} are duplicated".format(", ".join(intersect_keys)))
        x_index_columns = x.index.names
        y_index_columns = y.index.names
        if x_index_columns != y_index_columns:
            raise ValueError("All indices must be indexed by the same columns")
        index = cls()  # Create empty index
        index.surveys_dict = {**x.surveys_dict, **y.surveys_dict}
        index.headers = pd.merge(x.headers.reset_index(), y.headers.reset_index(), *args, **kwargs)
        index.headers.set_index(x_index_columns, inplace=True)
        index._index = index.headers.index.unique()
        if not len(index._index):
            raise ValueError("Empty index after merge")
        return index

    @classmethod
    def merge(cls, surveys, *args, **kwargs):
        survey_indices = cls.surveys_to_indices(surveys)
        index = reduce(lambda x, y: cls.merge_two_indices(x, y, *args, **kwargs), survey_indices)
        index._pos = index.build_pos()  # Build _pos dict explicitly if merge was called outside __init__
        return index

    @classmethod
    def concat(cls, surveys, *args, **kwargs):
        surveys = cls.surveys_to_indices(surveys)
        # TODO: concat
        index = None
        # index._pos = index.build_pos()  # Build _pos dict explicitly if concat was called outside __init__
        return index

    def create_subset(self, index):
        return type(self)(index=index, survey_dict=self.surveys_dict, headers=self.headers)

    def get_gather(self, survey_name, index):
        if survey_name not in self.surveys_dict:
            err_msg = "Unknown survey name {}, the index contains only {}"
            raise KeyError(err_msg.format(survey_name, ",".join(self.surveys_dict.keys())))
        concat_id = index[0]
        survey_index = index[1:]
        if len(survey_index) == 1:
            survey_index = survey_index[0]
        return self.surveys_dict[survey_name][concat_id].get_gather(index=survey_index)
