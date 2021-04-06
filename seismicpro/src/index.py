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
        if isinstance(index, SeismicIndex):
            index = index.index
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
        if mode in {"merge", "m"}:
            index = SeismicIndex.merge(surveys, *args, **kwargs)
        elif mode in {"concat", "c"}:
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
        # TODO: Mention in docs that only single index is allowed!
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
        # Pad lists in surveys_dict to max len for further concat to work correctly
        max_len = max(len(surveys) for surveys in index.surveys_dict.values())
        for survey in index.surveys_dict:
            index.surveys_dict[survey] += [None] * (max_len - len(index.surveys_dict[survey]))
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

    @staticmethod
    def reindex_headers(survey, start_concat_id):
        headers = survey.headers.copy()
        concat_id_values = headers.index.levels[headers.index.names.index("CONCAT_ID")] + start_concat_id
        headers.index = headers.index.set_levels(concat_id_values, "CONCAT_ID")
        start_concat_id += max(len(val) for val in survey.surveys_dict.values())
        return headers, start_concat_id

    @classmethod
    def concat(cls, surveys, *args, **kwargs):
        surveys = cls.surveys_to_indices(surveys)
        survey_names = surveys[0].surveys_dict.keys()
        if any(survey_names != survey.surveys_dict.keys() for survey in surveys):
            raise ValueError("Only surveys with the same name can be merged")

        headers_list = []
        start_concat_id = 0
        for survey in surveys:
            headers, start_concat_id = cls.reindex_headers(survey, start_concat_id)
            headers_list.append(headers)

        index = cls()  # Create empty index
        index.surveys_dict = {key: sum([survey.surveys_dict[key] for survey in surveys], []) for key in survey_names}
        index.headers = pd.concat(headers_list, *args, **kwargs)
        index._index = index.headers.index.unique()
        index._pos = index.build_pos()  # Build _pos dict explicitly if concat was called outside __init__
        return index

    def create_subset(self, index):
        return type(self)(index=index, survey_dict=self.surveys_dict, headers=self.headers)

    def get_gather(self, survey_name, index, **kwargs):
        if survey_name not in self.surveys_dict:
            err_msg = "Unknown survey name {}, the index contains only {}"
            raise KeyError(err_msg.format(survey_name, ",".join(self.surveys_dict.keys())))
        concat_id = index[0]
        survey_index = index[1:]
        if len(survey_index) == 1:
            survey_index = survey_index[0]
        return self.surveys_dict[survey_name][concat_id].get_gather(index=survey_index, combined=False, **kwargs)

    def get_combined_gather(self, survey_name, indices, **kwargs):
        gathers = []
        indices_headers = self.headers.loc[indices]
        for concat_id, sub_headers in indices_headers.groupby("CONCAT_ID"):
            concat_indices = sub_headers.reset_index(level=0, drop=True).index
            gathers.append(self.surveys_dict[survey_name][concat_id].get_gather(index=concat_indices,
                                                                                combined=True,
                                                                                **kwargs))
        return gathers

    def reindex(self, new_index):
        # We always keep 'CONCAT_ID' column in the index.
        self.headers.reset_index(level=self.headers.index.names[1:], inplace=True)
        self.headers.set_index(new_index, append=True, inplace=True)

        for surveys in self.surveys_dict.values():
            for survey in surveys:
                survey.reindex(new_index)

        self._index = self.headers.index.unique()
        self._pos = self.build_pos()  # Build _pos dict explicitly if concat was called outside __init__
        return self
