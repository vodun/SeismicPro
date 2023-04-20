"""Defines base containers - mixin classes that implement properties and basic processing logic for objects that store
particular types of data:
* `SamplesContainer` - implements extra properties for subclasses with defined `samples` attribute,
* `TraceContainer` - implements properties and processing methods for subclasses with defined `headers` attribute,
* `GatherContainer` - a subclass of `TraceContainer` that also implements fast selection of gather headers by index.
"""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl

from .decorators import batch_method
from .utils import to_list, get_cols, create_indexer, maybe_copy


class SamplesContainer:
    """A mixin class that implements extra properties for concrete subclasses with defined `samples` attribute that
    stores recording times for each trace value as a 1d `np.ndarray`."""

    @property
    def times(self):
        """1d np.ndarray of floats: Recording time for each trace value. Measured in milliseconds."""
        return self.samples

    @property
    def n_samples(self):
        """int: Trace length in samples."""
        return len(self.samples)

    @property
    def n_times(self):
        """int: Trace length in samples."""
        return len(self.times)


class TraceContainer:
    """A mixin class that implements extra properties and processing methods for concrete subclasses with defined
    `headers` attribute that stores loaded trace headers as a `pd.DataFrame`."""

    @property
    def indexed_by(self):
        """str or list of str: Names of header indices."""
        index_names = list(self.headers.index.names)
        if len(index_names) == 1:
            return index_names[0]
        return index_names

    @property
    def available_headers(self):
        """set of str: Names of available trace headers: both loaded and created manually."""
        return set(self.headers.columns) | set(self.headers.index.names)

    @property
    def n_traces(self):
        """int: The number of traces."""
        return len(self.headers)

    def __getitem__(self, key):
        """Select values of trace headers by their names and return them as a `np.ndarray`. Unlike `pandas` indexing
        allows for selection of headers the container is indexed by. The returned array will be 1d if a single header
        is selected and 2d otherwise.

        Parameters
        ----------
        key : str or list of str
            Names of headers to get values for.

        Returns
        -------
        result : np.ndarray
            Headers values.
        """
        return get_cols(self.headers, key)

    def __setitem__(self, key, value):
        """Set given values to selected headers.

        Parameters
        ----------
        key : str or list of str
            Headers to set values for.
        value : np.ndarray
            Headers values to set.
        """
        self.headers[key] = value

    def get_headers(self, cols):
        """Select values of trace headers by their names and return them as a `pandas.DataFrame`. Unlike `pandas`
        indexing allows for selection of headers the container is indexed by.

        Parameters
        ----------
        cols : str or list of str
            Names of headers to get values for.

        Returns
        -------
        result : pandas.DataFrame
            Headers values.
        """
        return pd.DataFrame(self[cols], columns=to_list(cols))

    def copy(self, ignore=None):
        """Perform a deepcopy of all attributes of `self` except for those specified in `ignore`, which are kept
        unchanged.

        Parameters
        ----------
        ignore : str or array-like of str, defaults to None
            Attributes that won't be copied.

        Returns
        -------
        copy : same type as self
            Copy of `self`.
        """
        ignore = set() if ignore is None else set(to_list(ignore))
        ignore_attrs = [getattr(self, attr) for attr in ignore]

        # Construct a memo dict with attributes, that should not be copied
        memo = {id(attr): attr for attr in ignore_attrs}
        return deepcopy(self, memo)

    @staticmethod
    def _apply(func, df, axis, unpack_args, **kwargs):
        """Apply a function to a `pd.DataFrame` along the specified axis.

        Parameters
        ----------
        func : callable
            A function to be applied to `df`.
        df : pd.DataFrame
            A `DataFrame` to which the function will be applied.
        axis : {0 or "index", 1 or "columns", None}
            An axis along which the function is applied:
            - 0 or "index": apply a function to each column,
            - 1 or "columns": apply a function to each row,
            - `None`: apply a function to the `DataFrame` as a whole.
        unpack_args : bool
            If `True`, row or column values are passed to `func` as individual arguments, otherwise the whole array is
            passed as a single arg. If `axis` is `None` and `unpack_args` is `True`, columns of the `df` are passed to
            the `func` as individual arguments.
        kwargs : misc, optional
            Additional keyword arguments to be passed to `func` or `pd.DataFrame.apply`.

        Returns
        -------
        result : np.ndarray
            The result of applying `func` to `df`.
        """
        if axis is None:
            args = (col_val for _, col_val in df.items()) if unpack_args else (df,)
            res = func(*args, **kwargs)
        else:
            # FIXME: Workaround for a pandas bug https://github.com/pandas-dev/pandas/issues/34822
            # raw=True causes incorrect apply behavior when axis=1 and multiple values are returned from `func`
            raw = axis != 1

            apply_func = (lambda args, **kwargs: func(*args, **kwargs)) if unpack_args else func
            res = df.apply(apply_func, axis=axis, raw=raw, result_type="expand", **kwargs)

        # Convert np.ndarray/pd.Series/pd.DataFrame outputs from `func` to a 2d array
        return pd.DataFrame(res).to_numpy()

    def _post_filter(self, mask):
        """Implement extra filtering logic of concrete subclass attributes if some of them should also be filtered
        besides `headers`."""
        _ = mask
        return

    @batch_method(target="for")
    def filter(self, cond, cols, axis=None, unpack_args=False, inplace=False, **kwargs):
        """Keep only those rows of `headers` where `cond` is `True`.

        Examples
        --------
        Keep only traces whose offset is less than 1500 meters:
        >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
        >>> survey.filter(lambda offset: offset < 1500, cols="offset", inplace=True)

        Parameters
        ----------
        cond : callable
            A function to be applied to `self.headers` to get a filtering mask. Must return a boolean array whose
            length equals to the length of `headers` and `True` values correspond to traces to keep.
        cols : str or list of str
            `self.headers` columns for which condition is checked.
        axis : {0 or "index", 1 or "columns", None}, optional, defaults to None
            An axis along which `cond` is applied:
            - 0 or "index": apply `cond` to each column,
            - 1 or "columns": apply `cond` to each row,
            - `None`: apply `cond` to the `DataFrame` as a whole.
        unpack_args : bool, optional, defaults to False
            If `True`, row or column values are passed to `cond` as individual arguments, otherwise the whole array is
            passed as a single arg. If `axis` is `None` and `unpack_args` is `True`, each column from `cols` is passed
            to the `cond` as an individual argument.
        inplace : bool, optional, defaults to False
            Whether to perform filtering inplace or process a copy.
        kwargs : misc, optional
            Additional keyword arguments to be passed to `cond` or `pd.DataFrame.apply`.

        Returns
        -------
        result : same type as self
            Filtered `self`.

        Raises
        ------
        ValueError
            If `cond` returns more than one bool value for each row of `headers`.
        """
        self = maybe_copy(self, inplace, ignore="headers")  # pylint: disable=self-cls-assignment
        cols = to_list(cols)
        headers = self.get_headers(cols)
        mask = self._apply(cond, headers, axis=axis, unpack_args=unpack_args, **kwargs)
        if (mask.ndim != 2) or (mask.shape[1] != 1):
            raise ValueError("cond must return a single value for each header row")
        if mask.dtype != np.bool_:
            raise ValueError("cond must return a bool value for each header row")
        mask = mask[:, 0]
        # Guarantee that a copy is set
        self.headers = self.headers.loc[mask].copy()  # pylint: disable=attribute-defined-outside-init
        if len(self.headers) == 0:
            warnings.warn("Empty headers after filtering", RuntimeWarning)
        self._post_filter(mask)
        return self

    @batch_method(target="for")
    def apply(self, func, cols, res_cols=None, axis=None, unpack_args=False, inplace=False, **kwargs):
        """Apply a function to `self.headers` along the specified axis.

        Examples
        --------
        Convert signed offsets to their absolute values:
        >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
        >>> survey.apply(lambda offset: np.abs(offset), cols="offset", inplace=True)

        Parameters
        ----------
        func : callable
            A function to be applied to `self.headers`. Must return a 2d object with shape (`len(self.headers)`,
            `len(res_cols)`).
        cols : str or list of str
            `self.headers` columns for which the function is applied.
        res_cols : str or list of str, optional, defaults to None
            `self.headers` columns in which the result is saved. If not given, equals to `cols`.
        axis : {0 or "index", 1 or "columns", None}, optional, defaults to None
            An axis along which the function is applied:
            - 0 or "index": apply a function to each column,
            - 1 or "columns": apply a function to each row,
            - `None`: apply a function to the `DataFrame` as a whole.
        unpack_args : bool, optional, defaults to False
            If `True`, row or column values are passed to `func` as individual arguments, otherwise the whole array is
            passed as a single arg. If `axis` is `None` and `unpack_args` is `True`, each column from `cols` is passed
            to the `func` as an individual argument.
        inplace : bool, optional, defaults to False
            Whether to apply the function inplace or to a copy.
        kwargs : misc, optional
            Additional keyword arguments to be passed to `func` or `pd.DataFrame.apply`.

        Returns
        -------
        result : same type as self
            `self` with the function applied.
        """
        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        cols = to_list(cols)
        headers = self.get_headers(cols)
        res_cols = cols if res_cols is None else to_list(res_cols)
        res = self._apply(func, headers, axis=axis, unpack_args=unpack_args, **kwargs)
        self.headers[res_cols] = res
        return self

    def load_headers(self, path, names=None, index_col=None, format="fwf", sep=None, usecols=None, skiprows=None, # pylint: disable=too-many-arguments
                     engine="pyarrow", decimal=None, encoding="UTF-8", keep_all_headers=False, inplace=False,
                     **kwargs):
        """"""
        self = maybe_copy(self, inplace, ignore="headers")  # pylint: disable=self-cls-assignment
        # TODO: can be infered from file as decimal if needed
        if sep is None:
            if format == "fwf":
                sep = r"\s+"
                engine = None
            elif format == "csv":
                sep = ","
            else:
                raise ValueError()

        # If decimal is not provided, try inferring it from the file
        if decimal is None:
            with open(path, 'r', encoding=encoding) as f:
                row = f.readline() if skiprows is None else [next(f) for _ in range(skiprows+1)][-1]
            decimal = '.' if '.' in row else ','

        if usecols is not None:
            usecols = np.asarray(usecols)
            if any(usecols < 0):
                sep = sep if format == "csv" else None
                with open(path, 'r', encoding=encoding) as f:
                    n_cols = len(f.readline().split(sep))
                usecols[usecols < 0] = n_cols + usecols[usecols < 0]

        loaded_df = pd.read_csv(path, sep=sep, names=names, index_col=index_col, usecols=usecols, decimal=decimal,
                                engine=engine, skiprows=skiprows, encoding=encoding, **kwargs)
        return loaded_df
        how = "left" if keep_all_headers else "inner"
        self.headers = self.headers.join(loaded_df, on=index_col, how=how, rsuffix="_loaded")

        if self.is_empty:
            warnings.warn("Empty headers after headers loading", RuntimeWarning)
        return self

    def dump_headers(self, path, columns, format="fwf", sep=',', col_space=8, dump_col_names=True, **kwargs):
        dump_df = self.get_headers(columns)
        if format == "fwf":
            dump_df.to_string(path, col_space=col_space, header=dump_col_names, index=False, **kwargs)
        elif format == "csv":
            dump_df = pl.from_pandas(dump_df)
            dump_df.write_csv(path, has_header=dump_col_names, separator=sep, **kwargs)
        else:
            raise ValueError(f"Unknown format {format}, avaliable formats are ['fwf', 'csv']")


class GatherContainer(TraceContainer):
    """A mixin class that implements extra properties and processing methods for concrete subclasses with defined
    `headers` attribute that stores loaded trace headers for several gathers as a `pd.DataFrame` and means for fast
    selection of gather headers by index."""

    def __len__(self):
        """The number of gathers."""
        return self.n_gathers

    def __contains__(self, index):
        """Returns whether a gather with given `index` is presented in `headers`."""
        return index in self.indices

    @property
    def headers(self):
        """pd.DataFrame: loaded trace headers."""
        return self._headers

    @headers.setter
    def headers(self, headers):
        """Reconstruct an indexer on each headers assignment."""
        if not (headers.index.is_monotonic_increasing or headers.index.is_monotonic_decreasing):
            headers = headers.sort_index(kind="stable")
        self._indexer = create_indexer(headers.index)
        self._headers = headers

    @property
    def indices(self):
        """pd.Index: indices of gathers."""
        return self._indexer.unique_indices

    @property
    def n_gathers(self):
        """int: The number of gathers."""
        return len(self.indices)

    @property
    def is_empty(self):
        """bool: Whether no gathers are stored in the container."""
        return self.n_gathers == 0

    def get_traces_locs(self, indices):
        """Get positions of traces in `headers` by `indices` of their gathers.

        Parameters
        ----------
        indices : array-like
            Indices of gathers to get trace locations for.

        Returns
        -------
        locations : array-like
            Locations of traces of the requested gathers.
        """
        return self._indexer.get_locs_in_indices(indices)

    def get_gathers_locs(self, indices):
        """Get ordinal positions of gathers in the container by their `indices`.

        Parameters
        ----------
        indices : array-like
            Indices of gathers to get ordinal positions for.

        Returns
        -------
        locations : np.ndarray
            Locations of the requested gathers.
        """
        return self._indexer.get_locs_in_unique_indices(indices)

    def get_headers_by_indices(self, indices):
        """Return headers for gathers with given `indices`.

        Parameters
        ----------
        indices : array-like
            Indices of gathers to get headers for.

        Returns
        -------
        headers : pd.DataFrame
            Selected headers values.
        """
        return self.headers.iloc[self.get_traces_locs(indices)]

    def copy(self, ignore=None):
        """Perform a deepcopy of all attributes of `self` except for indexer and those specified in `ignore`, which are
        kept unchanged.

        Parameters
        ----------
        ignore : str or array-like of str, defaults to None
            Attributes that won't be copied.

        Returns
        -------
        copy : same type as self
            Copy of `self`.
        """
        ignore = set() if ignore is None else set(to_list(ignore))
        return super().copy(ignore | {"_indexer"})

    def reindex(self, new_index, inplace=False):
        """Change the index of `self.headers` to `new_index`.

        Parameters
        ----------
        new_index : str or list of str
            Headers columns to become a new index.
        inplace : bool, optional, defaults to False
            Whether to perform reindexation inplace or return a new instance.

        Returns
        -------
        self : same type as self
            Reindexed self.
        """
        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        headers = self.headers
        headers.reset_index(inplace=True)
        headers.set_index(new_index, inplace=True)
        headers.sort_index(kind="stable", inplace=True)
        self.headers = headers
        return self
