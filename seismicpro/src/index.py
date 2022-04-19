"""Implements SeismicIndex class that allows for iteration over gathers in a survey or a group of surveys"""

import os
import warnings
from textwrap import indent, dedent
from functools import partial, reduce

import numpy as np
import pandas as pd

from .survey import Survey
from .containers import GatherContainer
from .utils import to_list
from ..batchflow import DatasetIndex


class IndexPart(GatherContainer):
    def __init__(self):
        self._headers = None
        self.indexer = None
        self.common_headers = set()
        self.surveys_dict = {}

    @property
    def survey_names(self):
        """list of str: names of surveys in the index part."""
        return sorted(self.surveys_dict.keys())

    @classmethod
    def from_attributes(cls, headers, surveys_dict, common_headers):
        part = cls()
        part.headers = headers
        part.common_headers = common_headers
        part.surveys_dict = surveys_dict
        return part

    @classmethod
    def from_survey(cls, survey, copy_headers=False):
        headers = survey.headers.copy(copy_headers)
        headers.columns = pd.MultiIndex.from_product([[survey.name], headers.columns])

        part = cls()
        part._headers = headers  # Avoid calling headers setter since the indexer is already calculated
        part.indexer = survey.indexer
        part.common_headers = set(headers.columns) - {"TRACE_SEQUENCE_FILE"}
        part.surveys_dict = {survey.name: survey}
        return part

    @staticmethod
    def _filter_equal(df, cols):
        drop_mask = reduce(np.logical_or, [np.ptp(df.loc[:, (slice(None), col)], axis=1) for col in cols])
        return df.loc[~drop_mask]

    def merge(self, other, on=None, validate="1:1"):
        self_indexed_by = set(to_list(self.indexed_by))
        other_indexed_by = set(to_list(other.indexed_by))
        if self_indexed_by != other_indexed_by:
            raise ValueError("All parts must be indexed by the same headers")
        if set(self.survey_names) & set(other.survey_names):
            raise ValueError("Only surveys with unique names can be merged")

        common_headers = self.common_headers & other.common_headers
        if on is None:
            on = common_headers
            left_df = self.headers
            right_df = other.headers
        else:
            on = set(to_list(on)) - self_indexed_by
            # Filter both self and other by equal values of on
            left_df = self._filter_equal(self.headers, on)
            right_df = self._filter_equal(other.headers, on)
        headers_to_check = common_headers - on

        merge_on = sorted(on)
        left_survey_name = self.survey_names[0]
        right_survey_name = other.survey_names[0]
        left_on = to_list(self.indexed_by) + [(left_survey_name, header) for header in merge_on]
        right_on = to_list(other.indexed_by) + [(right_survey_name, header) for header in merge_on]

        headers = pd.merge(left_df, right_df, how="inner", left_on=left_on, right_on=right_on, copy=True, sort=False,
                           validate=validate)

        # Recalculate common headers in the merged DataFrame
        common_headers = on | {header for header in headers_to_check
                                      if headers[left_survey_name, header].equals(headers[right_survey_name, header])}
        return self.from_attributes(headers, {**self.surveys_dict, **other.surveys_dict}, common_headers)

    def create_subset(self, indices):
        subset_headers = self.get_headers_by_indices(indices)
        return self.from_attributes(subset_headers, self.surveys_dict, self.common_headers)


class SeismicIndex(DatasetIndex):
    """A class that enumerates gathers in a survey or a group of surveys and allows iterating over them.

    While `Survey` describes a single SEG-Y file, `SeismicIndex` is primarily used to describe survey concatenation
    (e.g. when several fields are being processed in the same way one after another) or merging (e.g. when traces from
    the same field before and after a given processing stage must be matched and compared).

    TODO: finish docs

    Examples
    --------
    Let's consider 4 surveys, describing a single field before and after processing. Note that all of them have the
    same `header_index`:
    >>> s1_before = Survey(path, header_index=index_headers, name="before")
    >>> s2_before = Survey(path, header_index=index_headers, name="before")

    >>> s1_after = Survey(path, header_index=index_headers, name="after")
    >>> s2_after = Survey(path, header_index=index_headers, name="after")

    An index can be created from a single survey in the following way:
    >>> index = SeismicIndex(s1_before)

    If `s1_before` and `s2_before` represent different parts of the same field, they can be concatenated into one index
    to iterate over the whole field and process it at once. Both surveys must have the same `name`:
    >>> index = SeismicIndex(s1_before, s2_before, mode="c")

    Gathers before and after given processing stage can be matched using merge operation. Both surveys must have
    different `name`s:
    >>> index = SeismicIndex(s1_before, s1_after, mode="m")

    Merge can follow concat and vice versa. A more complex case, covering both operations is shown below:
    >>> index_before = SeismicIndex(s1_before, s2_before, mode="c")
    >>> index_after = SeismicIndex(s1_after, s2_after, mode="c")
    >>> index = SeismicIndex(index_before, index_after, mode="m")

    Parameters
    ----------
    args : tuple of Survey, IndexPart or SeismicIndex
        A sequence of surveys, indices or parts to construct an index.
    mode : {"c", "concat", "m", "merge", None}, optional, defaults to None
        A mode used to combine multiple `args` into a single index. If `None`, only one positional argument can be
        passed.
    copy_headers : bool, optional, defaults to False
        Whether to copy a `DataFrame` of trace headers while constructing index parts.
    kwargs : misc, optional
        Additional keyword arguments to :func:`~SeismicIndex.merge` if the corresponding mode was chosen.

    Attributes
    ----------
    parts : tuple of IndexPart
        Parts of the constructed index.
    index : tuple of pd.Index
        Unique identifiers of seismic gathers in each part of the index.
    """
    def __init__(self, *args, mode=None, copy_headers=False, **kwargs):
        self.parts = tuple()
        super().__init__(*args, mode=mode, copy_headers=copy_headers, **kwargs)

    @property
    def n_parts(self):
        """int: The number of parts in the index."""
        return len(self.parts)

    @property
    def n_gathers_by_part(self):
        """int: The number of gathers in each part of the index."""
        return [part.n_gathers for part in self.parts]

    @property
    def n_gathers(self):
        """int: The number of gathers in the index."""
        return sum(self.n_gathers_by_part)

    @property
    def n_traces_by_part(self):
        """int: The number of traces in each part of the index."""
        return [part.n_traces for part in self.parts]

    @property
    def n_traces(self):
        """int: The number of traces in the index."""
        return sum(self.n_traces_by_part)

    @property
    def indexed_by(self):
        """str or list of str or None: Names of header indices of each part. `None` for empty index."""
        if self.parts:
            return self.parts[0].indexed_by
        return None

    @property
    def survey_names(self):
        """list of str or None: Names of surveys in the index. `None` for empty index."""
        if self.parts:
            return self.parts[0].survey_names
        return None

    @property
    def is_empty(self):
        """bool: Whether the index is empty."""
        return self.n_parts == 0

    def __len__(self):
        """The number of gathers in the index."""
        return self.n_gathers

    def get_index_info(self, index_path="index", indent_size=0):
        """Recursively fetch index description string from the index itself and all the nested subindices."""
        if self.is_empty:
            return "Empty index"

        info_df = pd.DataFrame({"Gathers": self.n_gathers_by_part, "Traces": self.n_traces_by_part},
                               index=pd.RangeIndex(self.n_parts, name="Part"))
        for sur in self.survey_names:
            info_df[f"Survey {sur}"] = [os.path.basename(part.surveys_dict[sur].path) for part in self.parts]

        msg = f"""
        {index_path} info:

        Indexed by:                {", ".join(to_list(self.indexed_by))}
        Number of gathers:         {self.n_gathers}
        Number of traces:          {self.n_traces}
        Is split:                  {self.is_split}

        Index parts info:
        """
        msg = indent(dedent(msg) + info_df.to_string() + "\n", " " * indent_size)

        # Recursively fetch info about index splits
        for split_name in ("train", "test", "validation"):
            split = getattr(self, split_name)
            if split is not None:
                msg += "_" * 79 + "\n" + split.get_index_info(f"{index_path}.{split_name}", indent_size+4)
        return msg

    def __str__(self):
        """Print index metadata including information about its parts and underlying surveys."""
        msg = self.get_index_info()
        for i, part in enumerate(self.parts):
            for sur in part.survey_names:
                msg += "_" * 79 + "\n\n" + f"Part {i}, Survey {sur}\n\n" + str(part.surveys_dict[sur]) + "\n"
        return msg.strip()

    def info(self):
        """Print index metadata including information about its parts and underlying surveys."""
        print(self)

    #------------------------------------------------------------------------#
    #                         Index creation methods                         #
    #------------------------------------------------------------------------#

    @classmethod
    def _args_to_indices(cls, *args, copy_headers=False):
        indices = []
        for arg in args:
            if isinstance(arg, Survey):
                builder = cls.from_survey
            elif isinstance(arg, IndexPart):
                builder = cls.from_parts
            elif isinstance(arg, SeismicIndex):
                builder = cls.from_index
            else:
                raise ValueError(f"Unsupported type {type(arg)} to convert to index")
            indices.append(builder(arg, copy_headers=copy_headers))
        return indices

    @classmethod
    def _combine_indices(cls, *indices, mode=None, **kwargs):
        builders_dict = {
            "m": cls.merge,
            "merge": cls.merge,
            "c": partial(cls.concat, copy_headers=False),
            "concat": partial(cls.concat, copy_headers=False),
        }
        if mode not in builders_dict:
            raise ValueError(f"Unknown mode {mode}")
        return builders_dict[mode](*indices, **kwargs)

    def build_index(self, *args, mode=None, copy_headers=False, **kwargs):
        # Create an empty index if no args are given
        if not args:
            return tuple()

        # Don't copy headers if args are merged since pandas.merge will return a copy further
        if mode in {"m", "merge"}:
            copy_headers = False

        # Convert all args to SeismicIndex and combine them into a single index
        args = self._args_to_indices(*args, copy_headers=copy_headers)
        index = args[0] if len(args) == 1 else self._combine_indices(*args, mode=mode, **kwargs)

        # Copy parts from the created index to self and return gather indices for each part
        self.parts = index.parts
        return index.index

    @classmethod
    def from_parts(cls, *parts, copy_headers=False):
        survey_names = parts[0].survey_names
        if any(survey_names != part.survey_names for part in parts[1:]):
            raise ValueError("Only parts with the same survey names can be concatenated into one index")

        indexed_by = parts[0].indexed_by
        if any(indexed_by != part.indexed_by for part in parts[1:]):
            raise ValueError("All parts must be indexed by the same columns")

        if copy_headers:
            # TODO: copy headers
            parts = parts

        index = cls()
        index.parts = parts
        index._index = tuple(part.indices for part in parts)
        index.reset("iter")
        return index

    @classmethod
    def from_survey(cls, survey, copy_headers=False):
        return cls.from_parts(IndexPart.from_survey(survey, copy_headers=copy_headers))

    @classmethod
    def from_index(cls, index, copy_headers=False):
        if not copy_headers:
            return index
        # TODO: copy headers
        return index

    @classmethod
    def concat(cls, *args, copy_headers=False):
        indices = cls._args_to_indices(*args, copy_headers=copy_headers)
        parts = sum([index.parts for index in indices], tuple())
        return cls.from_parts(*parts, copy_headers=False)

    @classmethod
    def merge(cls, *args, **kwargs):
        indices = cls._args_to_indices(*args, copy_headers=False)
        if len({ix.n_parts for ix in indices}) != 1:
            raise ValueError("All indices being merged must have the same number of parts")
        ix_parts = [ix.parts for ix in indices]
        merged_parts = [reduce(lambda x, y: x.merge(y, **kwargs), parts) for parts in zip(*ix_parts)]

        # Warn if the whole index or some of its parts are empty
        empty_parts = [i for i, part in enumerate(merged_parts) if not part]
        if len(empty_parts) == len(merged_parts):
            warnings.warn("Empty index after merge", RuntimeWarning)
        elif empty_parts:
            warnings.warn(f"Empty parts {empty_parts} after merge", RuntimeWarning)

        return cls.from_parts(*merged_parts, copy_headers=False)

    #------------------------------------------------------------------------#
    #                 DatasetIndex interface implementation                  #
    #------------------------------------------------------------------------#

    def build_pos(self):
        """Implement degenerate `get_pos` to decrease computational complexity since `SeismicIndex` provides its own
        interface to get gathers from each of its parts."""
        return None

    def index_by_pos(self, pos):
        """Return gather index and part by its position in the index.

        Parameters
        ----------
        pos : int
            Ordinal number of the gather in the index.

        Returns
        -------
        index : int or tuple
            Gather index.
        part : int
            Index part to get the gather from.
        """
        part_pos_borders = np.cumsum([0] + self.n_gathers_by_part)
        part = np.searchsorted(part_pos_borders[1:], pos, side="right")
        return self.indices[part][pos - part_pos_borders[part]], part

    def subset_by_pos(self, pos):
        """Return a subset of gather indices by their positions in the index.

        Parameters
        ----------
        pos : int or array-like of int
            Ordinal numbers of gathers in the index.

        Returns
        -------
        indices : list of pd.Index
            Gather indices of the subset by each index part.
        """
        pos = np.sort(np.atleast_1d(pos))
        part_pos_borders = np.cumsum([0] + self.n_gathers_by_part)
        pos_by_part = np.split(pos, np.searchsorted(pos, part_pos_borders[1:]))
        part_indices = [part_pos - part_start for part_pos, part_start in zip(pos_by_part, part_pos_borders[:-1])]
        return tuple(index[subset] for index, subset in zip(self.index, part_indices))

    def create_subset(self, index):
        """Return a new index object based on a subset of its indices given.

        Parameters
        ----------
        index : SeismicIndex or tuple of pd.Index
            Gather indices of the subset to create a new `SeismicIndex` object for. If `tuple` of `pd.Index`, each item
            defines gather indices of the corresponding part in `self`.

        Returns
        -------
        subset : SeismicIndex
            A subset of the index.
        """
        if isinstance(index, SeismicIndex):
            index = index.index
        if len(index) != self.n_parts:
            raise ValueError("Index length must match the number of parts")
        return self.from_parts(*[part.create_subset(ix) for part, ix in zip(self.parts, index)], copy_headers=False)

    #------------------------------------------------------------------------#
    #                     Statistics computation methods                     #
    #------------------------------------------------------------------------#

    def collect_stats(self, n_quantile_traces=100000, quantile_precision=2, limits=None, bar=True):
        """Collect the following trace data statistics for each survey in the index or a dataset:
        1. Min and max amplitude,
        2. Mean amplitude and trace standard deviation,
        3. Approximation of trace data quantiles with given precision.

        Since fair quantile calculation requires simultaneous loading of all traces from the file we avoid such memory
        overhead by calculating approximate quantiles for a small subset of `n_quantile_traces` traces selected
        randomly. Only a set of quantiles defined by `quantile_precision` is calculated, the rest of them are linearly
        interpolated by the collected ones.

        After the method is executed all calculated values can be obtained via corresponding attributes of the surveys
        in the index and their `has_stats` flag is set to `True`.

        Examples
        --------
        Statistics calculation for the whole index can be done as follows:
        >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
        >>> index = SeismicIndex(survey).collect_stats()

        Statistics can be calculated for a dataset as well:
        >>> dataset = SeismicDataset(index).collect_stats()

        After a train-test split is performed, `train` and `test` refer to the very same `Survey` instances. This
        allows for `collect_stats` to be used to calculate statistics for the training set and then use them to
        normalize gathers from the testing set to avoid data leakage during machine learning model training:
        >>> dataset.split()
        >>> dataset.train.collect_stats()
        >>> dataset.test.next_batch(1).load(src="survey").scale_standard(src="survey", use_global=True)

        Note that if no gathers from a particular survey were included in the training set its stats won't be
        collected!

        Parameters
        ----------
        n_quantile_traces : positive int, optional, defaults to 100000
            The number of traces to use for quantiles estimation.
        quantile_precision : positive int, optional, defaults to 2
            Calculate an approximate quantile for each q with `quantile_precision` decimal places. All other quantiles
            will be linearly interpolated on request.
        limits : int or tuple or slice, optional
            Time limits to be used for statistics calculation. `int` or `tuple` are used as arguments to init a `slice`
            object. If not given, `limits` passed to `Survey.__init__` are used. Measured in samples.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        self : same type as self
            An index or a dataset with collected stats. Sets `has_stats` flag to `True` and updates statistics
            attributes inplace for each of the underlying surveys.
        """
        for part in self.parts:
            for sur in part.surveys_dict.values():
                sur.collect_stats(indices=part.indices, n_quantile_traces=n_quantile_traces,
                                  quantile_precision=quantile_precision, limits=limits, bar=bar)
        return self

    #------------------------------------------------------------------------#
    #                            Loading methods                             #
    #------------------------------------------------------------------------#

    def get_gather(self, index, part=None, survey_name=None, limits=None, copy_headers=False):
        if part is None and self.n_parts > 1:
            raise ValueError("part must be specified if the index is concatenated")
        if part is None:
            part = 0
        index_part = self.parts[part]

        if survey_name is None and len(self.survey_names) > 1:
            raise ValueError("survey_name must be specified if the index is merged")
        if survey_name is None:
            survey_name = self.survey_names[0]
        survey = index_part.surveys_dict[survey_name]

        gather_headers = index_part.get_headers_by_indices((index,))[survey_name]
        return survey.load_gather(headers=gather_headers, limits=limits, copy_headers=copy_headers)

    def sample_gather(self, part=None, survey_name=None, limits=None, copy_headers=False):
        if part is None:
            part_weights = np.array(self.n_gathers_by_part) / self.n_gathers
            part = np.random.choice(self.n_parts, p=part_weights)
        if survey_name is None:
            survey_name = np.random.choice(self.survey_names)
        index = np.random.choice(self.parts[part].indices)
        return self.get_gather(index, part, survey_name, limits=limits, copy_headers=copy_headers)
