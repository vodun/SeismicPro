import os
from copy import deepcopy
from itertools import chain
from functools import reduce
from textwrap import dedent, indent

import numpy as np
import pandas as pd

from .survey import Survey
from .utils import maybe_copy, unique_indices_sorted
from ..batchflow import DatasetIndex


class SeismicIndex(DatasetIndex):
    def __init__(self, index=None, surveys=None, mode=None, **kwargs):
        self.headers = None
        self.surveys_dict = None
        self.index_to_headers_pos = None
        super().__init__(index=index, surveys=surveys, mode=mode, **kwargs)

    @property
    def next_concat_id(self):
        return max(len(surveys) for surveys in self.surveys_dict.values())

    def _get_index_info(self, indents, prefix):
        groupped_headers = self.headers.index.to_frame(index=False).groupby(by="CONCAT_ID")
        data = [[name, len(group), len(group.drop_duplicates())] for name, group in groupped_headers]
        info_df = pd.DataFrame.from_records(data, columns=["CONCAT_ID", "Num Traces", "Num Gathers"],
                                            index='CONCAT_ID')

        for sur_name in self.surveys_dict.keys():
            for ix, sur in enumerate(self.surveys_dict[sur_name]):
                file_name = os.path.basename(sur.path) if sur is not None else None
                info_df.loc[ix, 'Survey ' + sur_name] = file_name
        info_df.sort_index(inplace=True)

        split_names = ['train', 'test', 'validation']
        split_indices = [(getattr(self, name), name) for name in split_names if getattr(self, name) is not None]

        msg = f"""
        {prefix} info:


        Index name(s):             {', '.join(self.headers.index.names)}
        Number of traces:          {np.nansum(info_df["Num Traces"])}
        Number of gathers:         {np.nansum(info_df["Num Gathers"])}
        Is split:                  {any(split_indices)}


        The table describes surveys contained in the index:
        """
        msg = dedent(msg) + info_df.to_string() + '\n'

        nested_indices_msg = ""
        for index, name in split_indices:
            index_msg = index._get_index_info(indents=indents + '    ', prefix=prefix + '.' + name)
            nested_indices_msg += f"\n{'_'*79}" + index_msg
        return indent(msg, indents) + nested_indices_msg

    def __str__(self):
        return self._get_index_info(indents='', prefix="index")

    def info(self):
        print(self)

    #------------------------------------------------------------------------#
    #                         Index creation methods                         #
    #------------------------------------------------------------------------#

    def build_index(self, index=None, surveys=None, mode=None, **kwargs):
        # Create an empty index if both index and surveys are not specified
        if index is None and surveys is None:
            return None

        # If survey is passed, choose index builder depending on given mode
        if surveys is not None:
            builders_dict = {
                "m": SeismicIndex.merge,
                "c": SeismicIndex.concat,
                "merge": SeismicIndex.merge,
                "concat": SeismicIndex.concat,
                None: SeismicIndex.from_survey,
            }
            if mode not in builders_dict:
                raise ValueError("Unknown mode {}".format(mode))
            index = builders_dict[mode](surveys, **kwargs)

        # Check that passed or created index has SeismicIndex type
        if not isinstance(index, SeismicIndex):
            raise ValueError(f"SeismicIndex instance is expected as an index, but {type(index)} was given")

        # Copy internal attributes from passed or created index into self
        self.headers = index.headers
        self.surveys_dict = index.surveys_dict
        self.index_to_headers_pos = index.index_to_headers_pos
        return index.index

    @classmethod
    def from_attributes(cls, index=None, index_to_headers_pos=None, headers=None, surveys_dict=None):
        # Create empty SeismicIndex to fill with given headers and surveys_dict
        new_index = cls()
        new_index.headers = headers
        new_index.surveys_dict = surveys_dict

        if index is None:
            # Calculate unique header indices. This approach is way faster, than pd.unique or np.unique since we
            # guarantee that the index of the header is monotonic
            if not headers.index.is_monotonic_increasing:
                raise ValueError("Headers index must be monotonically increasing")
            unique_header_indices = unique_indices_sorted(headers.index.to_frame().values)
            index = headers.index[unique_header_indices]

            # Calculate a mapping from index value to its position in headers to further speed up create_subset
            index_to_headers_pos = {}
            start_pos = unique_header_indices
            end_pos = chain(unique_header_indices[1:], [len(headers)])
            for ix, start, end in zip(index, start_pos, end_pos):
                index_to_headers_pos[ix] = range(start, end)

        # Set _index explicitly since already created index is modified
        new_index._index = index
        new_index.index_to_headers_pos = index_to_headers_pos
        return new_index

    @classmethod
    def from_survey(cls, survey):
        if not isinstance(survey, Survey):
            raise ValueError(f"Survey instance is expected, but {type(survey)} was given. "
                              "Probably you forgot to specify mode")

        # Copy headers from survey and create zero CONCAT_ID column as the first index level
        headers = survey.headers.copy()
        old_index = headers.index.names
        headers.reset_index(inplace=True)
        headers["CONCAT_ID"] = 0
        headers.set_index(["CONCAT_ID"] + old_index, inplace=True)

        surveys_dict = {survey.name: [survey]}
        return cls.from_attributes(headers=headers, surveys_dict=surveys_dict)

    @staticmethod
    def _surveys_to_indices(surveys):
        survey_indices = []
        for survey in surveys:
            if not isinstance(survey, SeismicIndex):
                if not isinstance(survey, Survey):
                    raise ValueError("Each survey must have either Survey or SeismicIndex type, "
                                     f"but {type(survey)} was given")
                survey = SeismicIndex(surveys=survey)
            survey_indices.append(survey)
        return survey_indices

    @classmethod
    def _merge_two_indices(cls, x, y, **kwargs):
        intersect_keys = x.surveys_dict.keys() & y.surveys_dict.keys()
        if intersect_keys:
            raise ValueError("Only surveys with unique names can be merged, "
                             "but {} are duplicated".format(", ".join(intersect_keys)))

        x_index_columns = x.index.names
        y_index_columns = y.index.names
        if x_index_columns != y_index_columns:
            raise ValueError("All indices must be indexed by the same columns")

        headers = pd.merge(x.headers.reset_index(), y.headers.reset_index(), **kwargs)
        if headers.empty:
            raise ValueError("Empty index after merge")

        surveys_dict = {**x.surveys_dict, **y.surveys_dict}
        max_len = max(x.next_concat_id, y.next_concat_id)
        dropped_ids = np.setdiff1d(np.arange(max_len), np.unique(headers["CONCAT_ID"]))
        for survey in surveys_dict:
            # Pad lists in surveys_dict to max len for further concat to work correctly
            surveys_dict[survey] += [None] * (max_len - len(surveys_dict[survey]))
            # If some CONCAT_IDs were dropped after merge, set the corresponding survey values to None
            for concat_id in dropped_ids:
                surveys_dict[survey][concat_id] = None

        return cls.from_attributes(headers=headers.set_index(x_index_columns), surveys_dict=surveys_dict)

    @classmethod
    def merge(cls, surveys, **kwargs):
        indices = cls._surveys_to_indices(surveys)
        index = reduce(lambda x, y: cls._merge_two_indices(x, y, **kwargs), indices)
        return index

    @classmethod
    def concat(cls, surveys, **kwargs):
        indices = cls._surveys_to_indices(surveys)
        survey_names = indices[0].surveys_dict.keys()
        if any(survey_names != index.surveys_dict.keys() for index in indices):
            raise ValueError("Only surveys with the same names can be concatenated")

        index_columns = indices[0].headers.index.names
        if any(index_columns != index.headers.index.names for index in indices):
            raise ValueError("All indices must be indexed by the same columns")

        # Update CONCAT_ID values in all the indices to avoid collisions after concatenation
        headers_list = []
        concat_id_shift = 0
        for index in indices:
            headers = index.headers.copy()
            concat_id = headers.index.levels[0] + concat_id_shift
            headers.index = headers.index.set_levels(concat_id, "CONCAT_ID")

            headers_list.append(headers)
            concat_id_shift += index.next_concat_id

        headers = pd.concat(headers_list, **kwargs)
        surveys_dict = {survey_name: sum([index.surveys_dict[survey_name] for index in indices], [])
                        for survey_name in survey_names}
        return cls.from_attributes(headers=headers, surveys_dict=surveys_dict)

    #------------------------------------------------------------------------#
    #                 DatasetIndex interface implementation                  #
    #------------------------------------------------------------------------#

    def build_pos(self):
        return None

    def get_pos(self, index):
        # TODO: Mention in docs that only single index is allowed!
        return self.index.get_loc(index)

    def create_subset(self, index):
        if isinstance(index, SeismicIndex):
            index = index.index

        # Sort index of the subset. Otherwise subset.headers may become unsorted in pipelines with shuffle=True
        # resulting in non-working .loc
        index, _ = index.sortlevel()

        # Calculate positions of indices in header to perform .iloc instead of .loc, which is orders of magnitude
        # faster, and update index_to_headers_pos dict for further create_subset to work correctly
        headers_indices = []
        index_to_headers_pos = {}
        curr_index = 0
        for item in index:
            item_indices = self.index_to_headers_pos[item]
            headers_indices.append(item_indices)
            index_to_headers_pos[item] = range(curr_index, curr_index + len(item_indices))
            curr_index += len(item_indices)

        headers = self.headers.iloc[list(chain.from_iterable(headers_indices))]
        subset = self.from_attributes(index=index, index_to_headers_pos=index_to_headers_pos,
                                      headers=headers, surveys_dict=self.surveys_dict)
        return subset

    #------------------------------------------------------------------------#
    #                       Index manipulation methods                       #
    #------------------------------------------------------------------------#

    def copy(self, copy_surveys=True):
        if copy_surveys:
            return deepcopy(self)
        surveys_dict = self.surveys_dict
        self.surveys_dict = None
        self_copy = deepcopy(self)
        self_copy.surveys_dict = surveys_dict
        self.surveys_dict = surveys_dict
        return self_copy

    def get_gather(self, survey_name, concat_id, survey_index, **kwargs):
        if survey_name not in self.surveys_dict:
            err_msg = "Unknown survey name {}, the index contains only {}"
            raise KeyError(err_msg.format(survey_name, ", ".join(self.surveys_dict.keys())))
        survey = self.surveys_dict[survey_name][concat_id]
        gather_headers = self.headers.loc[concat_id].loc[survey_index]
        return survey.load_gather(headers=gather_headers, **kwargs)

    def reindex(self, new_index, inplace=False):
        self = maybe_copy(self, inplace)

        # Reindex headers, keeping CONCAT_ID column in it
        self.headers.reset_index(level=self.headers.index.names[1:], inplace=True)
        self.headers.set_index(new_index, append=True, inplace=True)
        self.headers.sort_index(inplace=True)

        # Set _index explicitly since already created index is modified
        # unique_indices_sorted is used since headers index is guaranteed to be sorted
        uniques_indices = unique_indices_sorted(headers.index.to_frame().values)
        self._index = headers.index[uniques_indices]

        # Reindex all the underlying surveys
        for surveys in self.surveys_dict.values():
            for survey in surveys:
                # None survey values can be created by _merge_two_indices
                if survey is not None:
                    survey.reindex(new_index, inplace=True)
        return self
