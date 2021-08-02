"""Implements SeismicIndex class that allows for iteration over gathers in a survey or a group of surveys"""

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
    """A class that enumerates gathers in a survey or a group of surveys and allows iterating over them.

    While `Survey` describes a single SEG-Y file, `SeismicIndex` is primarily used to describe survey concatenation
    (e.g. when several fields are being processed in the same way one after another) or merging (e.g. when traces from
    the same field before and after a given processing stage must be matched and compared).

    In order to enumerate seismic gathers from different surveys, an instance of `SeismicIndex` stores combined headers
    of all the surveys in its `headers` attribute and unique identifiers of all the resulting seismic gathers as a
    `pd.MultiIndex` in its `index` attribute.

    `SeismicIndex` object can be created either from a single survey or a list of surveys:
    1. A single survey is handled in a straightforward way: `headers` attribute of the resulting index is a copy of
       survey headers with an extra `CONCAT_ID` column with zero value added to its index to the left. This column is
       redundant in case of a single survey but allows for unambiguous survey identification when concatenation is
       performed.
    2. If several surveys are passed they must be created with the same `header_index`. First, they are independently
       converted to `SeismicIndex` as described above and then the `headers` attribute of the resulting index is built
       depending on the given `mode`:
        - "c" or "concat":
          All the surveys must have the same `name`. `CONCAT_ID` column is updated to be the ordinal number of a survey
          in the list for each of the created indices and then headers concatenation is performed via `pd.concat`. Here
          `CONCAT_ID` acts as a survey identifier since traces from different SEG-Y files may have the same headers
          making it impossible to recover a source survey for a trace with given headers.
        - "m" or "merge":
          All the surveys must have different `name` specified. In this case, `headers` are obtained by joining survey
          headers via `pd.merge`. By default, merging is performed by all the headers columns including `CONCAT_ID`
          allowing for several groups of concatenated surveys to be consequently merged.

    `SeismicIndex` keeps track of all the surveys used during its creation in a `surveys_dict` attribute with the
    following structure:
    `{survey_name_1: [survey_1_1, survey_1_2, survey_1_N],
      ...,
      survey_name_M: [survey_M_1, survey_M_2, survey_M_N],
     }`
    Dict keys here are the names of surveys being merged, while values are lists with the same length equal to the
    number of concatenated surveys.

    Thus, base survey to get a gather with given headers from is determined by both its name and `CONCAT_ID`. The
    gather itself can be obtained by calling :func:`~SeismicIndex.get_gather` method, while iteration over gathers is
    performed via :func:`~SeismicIndex.next_batch`.

    Examples
    --------
    Let's consider 4 surveys, describing a single field before and after processing. Note that all of them have the
    same `header_index`:
    >>> s1_before = Survey(path, header_index=index_headers, name="before")
    >>> s2_before = Survey(path, header_index=index_headers, name="before")

    >>> s1_after = Survey(path, header_index=index_headers, name="after")
    >>> s2_after = Survey(path, header_index=index_headers, name="after")

    An index can be created from a single survey in the following way:
    >>> index = SeismicIndex(surveys=s1_before)

    If `s1_before` and `s2_before` represent different parts of the same field, they can be concatenated into one index
    to iterate over the whole field and process it at once. Both surveys must have the same `name`:
    >>> index = SeismicIndex(surveys=[s1_before, s2_before], mode="c")

    Gathers before and after given processing stage can be matched using merge operation. Both surveys must have
    different `name`s:
    >>> index = SeismicIndex(surveys=[s1_before, s1_after], mode="m")

    Merge can follow concat and vice versa. A more complex case, covering both operations is shown below:
    >>> index_before = SeismicIndex(surveys=[s1_before, s2_before], mode="c")
    >>> index_after = SeismicIndex(surveys=[s1_after, s2_after], mode="c")
    >>> index = SeismicIndex(surveys=[index_before, index_after], mode="m")

    Parameters
    ----------
    index : SeismicIndex, optional
        Base index to use as is if no surveys were passed.
    surveys : Survey or list of Survey, optional
        Surveys to use to construct an index.
    mode : {"c", "concat", "m", "merge", None}, optional, defaults to None
        A mode used to combine multiple surveys into an index. If `None`, only a single survey can be passes to a
        `surveys` arg.
    kwargs : misc, optional
        Additional keyword arguments to index builder method for given `mode` (currently :func:`~SeismicIndex.merge` or
        :func:`~SeismicIndex.concat`).

    Attributes
    ----------
    headers : pd.DataFrame
        Combined headers of all the surveys used to create the index.
    surveys_dict : dict
        A dict, tracking surveys used to create the index. Its keys are the names of surveys being merged, while values
        are lists with the same length equal to the number of concatenated surveys.
    index : pd.MultiIndex
        Unique identifiers of seismic gathers in the constructed index.
    index_to_headers_pos : dict
        A mapping from an index value to a range of corresponding `headers` rows.
    """
    def __init__(self, index=None, surveys=None, mode=None, **kwargs):
        self.headers = None
        self.surveys_dict = None
        self.index_to_headers_pos = None
        super().__init__(index=index, surveys=surveys, mode=mode, **kwargs)

    @property
    def next_concat_id(self):
        """int: The number of concatenated surveys in the index."""
        return max(len(surveys) for surveys in self.surveys_dict.values())

    def _get_index_info(self, indents, prefix):
        """Recursively fetch index description string from the index itself and all the nested subindices."""
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
            # pylint: disable=protected-access
            index_msg = index._get_index_info(indents=indents + '    ', prefix=prefix + '.' + name)
            # pylint: enable=protected-access
            nested_indices_msg += f"\n{'_'*79}" + index_msg
        return indent(msg, indents) + nested_indices_msg

    def __str__(self):
        """Print index metadata including information about its surveys and total number of traces and gathers."""
        return self._get_index_info(indents='', prefix="index")

    def info(self):
        """Print index metadata including information about its surveys and total number of traces and gathers."""
        print(self)

    #------------------------------------------------------------------------#
    #                         Index creation methods                         #
    #------------------------------------------------------------------------#

    def build_index(self, index=None, surveys=None, mode=None, **kwargs):
        """Build an index from args in the following way:
        1. If both `index` and `surveys` are `None` an empty index is created.
        2. If `surveys` is given then index is created according to the `mode` specified.
        3. Otherwise passed `index` is used as is.

        Parameters
        ----------
        index : SeismicIndex, optional
            Base index to use as is if no surveys were passed.
        surveys : Survey or list of Survey, optional
            Surveys to use to construct an index.
        mode : {"c", "concat", "m", "merge", None}
            A mode used to combine multiple surveys into an index. If `None`, only a single survey can be passes to a
            `surveys` arg.
        kwargs : misc, optional
            Additional keyword arguments to index builder method for given `mode` (currently
            :func:`~SeismicIndex.merge` or :func:`~SeismicIndex.concat`).

        Returns
        -------
        index : pd.MultiIndex
            Unique identifiers of seismic gathers in the constructed index.

        Raises
        ------
        ValueError
            If unknown `mode` was passed.
        TypeError
            If `index` of a wrong type was passed.
        """
        # Create an empty index if both index and surveys are not specified
        if index is None and surveys is None:
            return None

        # If surveys are passed, choose index builder depending on given mode
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
            raise TypeError(f"SeismicIndex instance is expected as an index, but {type(index)} was given")

        # Copy internal attributes from passed or created index into self
        self.headers = index.headers
        self.surveys_dict = index.surveys_dict
        self.index_to_headers_pos = index.index_to_headers_pos
        return index.index

    @classmethod
    def from_attributes(cls, headers, surveys_dict, index=None, index_to_headers_pos=None):
        """Create a new `SeismicIndex` instance from its attributes.

        Parameters
        ----------
        headers : pd.DataFrame
            Headers of the index being created.
        surveys_dict : dict
            A dict of surveys used by the index. Its keys are the names of surveys being merged, while values are lists
            with the same length equal to the number of concatenated surveys.
        index : pd.MultiIndex, optional
            Unique identifiers of seismic gathers in the constructed index. If not given, calculated by passed
            `headers`.
        index_to_headers_pos : dict, optional
            A mapping from an index value to a range of corresponding `headers` rows. If not given, calculated by
            passed `headers`.

        Returns
        -------
        index : SeismicIndex
            Constructed index.

        Raises
        ------
        ValueError
            If `headers` index is not monotonically increasing.
        """
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
        """Create a new `SeismicIndex` instance from a single survey.

        `headers` attribute of the resulting index is a copy of survey `headers` with an extra `CONCAT_ID` column with
        zero value added to its index to the left.

        Parameters
        ----------
        survey : Survey
            A survey used to build an index.

        Returns
        -------
        index : SeismicIndex
            Constructed index.

        Raises
        ------
        TypeError
            If `survey` of a wrong type was passed.
        """
        if not isinstance(survey, Survey):
            raise TypeError(f"Survey instance is expected, but {type(survey)} was given. "
                             "Probably you forgot to specify the mode")

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
        """Cast each element of a list of `Survey` or `SeismicIndex` instances to a `SeismicIndex` type."""
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
        """Merge two `SeismicIndex` instances into one."""
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
        """Merge several surveys into a single index.

        All the surveys being merged must be created with the same `header_index`, but have different `name`s. First,
        they are independently converted to `SeismicIndex` and then the resulting index `headers` are calculated by
        joining the obtained headers via `pd.merge`. By default, merging is performed by all the columns including
        `CONCAT_ID` allowing for several groups of concatenated surveys to be consequently merged.

        Notes
        -----
        A detailed description of index merging can be found in :class:`~SeismicIndex` docs.

        Parameters
        ----------
        surveys : list of Survey
            A list of surveys to be merged.
        kwargs : misc, optional
            Additional keyword arguments to :func:`~pandas.merge`.

        Returns
        -------
        index : SeismicIndex
            Merged index.

        Raises
        ------
        ValueError
            If surveys with same names were passed.
            If survey headers are not indexed by the same columns.
            If an empty index was obtained after merging.
        """
        indices = cls._surveys_to_indices(surveys)
        index = reduce(lambda x, y: cls._merge_two_indices(x, y, **kwargs), indices)
        return index

    @classmethod
    def concat(cls, surveys, **kwargs):
        """Concatenate several surveys into a single index.

        All the surveys being concatenated must be created with the same `header_index`, and have the same `name`.
        First, they are independently converted to `SeismicIndex` and then `CONCAT_ID` column is updated to be the
        ordinal number of a survey in the list for each of the created indices. The resulting index `headers` are
        calculated by concatenating the obtained headers via `pd.concat`. `CONCAT_ID` acts as a survey identifier since
        traces from different SEG-Y files may have the same headers making it impossible to recover a source survey for
        a trace with given headers.

        Notes
        -----
        A detailed description of index concatenation can be found in :class:`~SeismicIndex` docs.

        Parameters
        ----------
        surveys : list of Survey
            A list of surveys to be concatenated.
        kwargs : misc, optional
            Additional keyword arguments to :func:`~pandas.concat`.

        Returns
        -------
        index : SeismicIndex
            Concatenated index.

        Raises
        ------
        ValueError
            If surveys with different names were passed.
            If survey headers are not indexed by the same columns.
        """
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
        """Implement degenerative `get_pos` to decrease computational complexity since `pd.MultiIndex` provides its own
        interface to get a position of a value in the index."""
        return None

    def get_pos(self, index):
        """Return a position of an item in the index.

        Notes
        -----
        Unlike `BatchFlow` `DatasetIndex.get_pos`, only a single index value is allowed.

        Parameters
        ----------
        index : tuple
            Multiindex value to return position of.

        Returns
        -------
        pos : int
            Position of the given item.
        """
        return self.index.get_loc(index)

    def create_subset(self, index):
        """Return a new index object based on the subset of indices given.

        Notes
        -----
        During the call subset of `self.headers` is calculated which may take a while for large indices.

        Parameters
        ----------
        index : SeismicIndex or pd.MultiIndex
            Index values of the subset to create a new `SeismicIndex` object for.

        Returns
        -------
        subset : SeismicIndex
            A subset of the index.
        """
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
        subset = self.from_attributes(headers=headers, surveys_dict=self.surveys_dict, index=index,
                                      index_to_headers_pos=index_to_headers_pos)
        return subset

    #------------------------------------------------------------------------#
    #                       Index manipulation methods                       #
    #------------------------------------------------------------------------#

    def copy(self, copy_surveys=True):
        """Perform a deepcopy of all index attributes except for `surveys_dict`, which will be copied only if
        `copy_surveys` is `True`.

        Parameters
        ----------
        copy_surveys : bool, optional, defaults to True
            Whether to deepcopy information about surveys in the index.

        Returns
        -------
        copy : SeismicIndex
            A copy of the index.
        """
        if copy_surveys:
            return deepcopy(self)
        surveys_dict = self.surveys_dict
        self.surveys_dict = None
        self_copy = deepcopy(self)
        self_copy.surveys_dict = surveys_dict
        self.surveys_dict = surveys_dict
        return self_copy

    def get_gather(self, survey_name, concat_id, gather_index, **kwargs):
        """Get a gather from a given survey by its index.

        Parameters
        ----------
        survey_name : str
            Survey name to get the gather from.
        concat_id : int
            Concatenation index of the source survey.
        gather_index : pd.MultiIndex
            Indices of gather traces to get.
        kwargs : misc, optional
            Additional keyword arguments to :func:`~Survey.load_gather`.

        Returns
        -------
        gather : Gather
            Loaded gather.

        Raises
        ------
        KeyError
            If unknown survey name was passed.
        """
        if survey_name not in self.surveys_dict:
            err_msg = "Unknown survey name {}, the index contains only {}"
            raise KeyError(err_msg.format(survey_name, ", ".join(self.surveys_dict.keys())))
        survey = self.surveys_dict[survey_name][concat_id]
        gather_headers = self.headers.loc[concat_id].loc[gather_index]
        return survey.load_gather(headers=gather_headers, **kwargs)

    def reindex(self, new_index, inplace=False):
        """Reindex `self` with new headers columns. All underlying surveys are reindexed as well.

        Parameters
        ----------
        new_index : str or list of str
            Headers columns to become a new index. Note, that `CONCAT_ID` is always preserved in the index and should
            not be specified.
        inplace : bool, optional, defaults to False
            Whether to perform reindexation inplace or return a new index instance.

        Returns
        -------
        index : SeismicIndex
            Reindexed `self`.
        """
        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment

        # Reindex headers, keeping CONCAT_ID column in it
        self.headers.reset_index(level=self.headers.index.names[1:], inplace=True)
        self.headers.set_index(new_index, append=True, inplace=True)
        self.headers.sort_index(inplace=True)

        # Set _index explicitly since already created index is modified
        # unique_indices_sorted is used since headers index is guaranteed to be sorted
        uniques_indices = unique_indices_sorted(self.headers.index.to_frame().values)
        self._index = self.headers.index[uniques_indices]

        # Reindex all the underlying surveys
        for surveys in self.surveys_dict.values():
            for survey in surveys:
                # None survey values can be created by _merge_two_indices
                if survey is not None:
                    survey.reindex(new_index, inplace=True)
        return self
