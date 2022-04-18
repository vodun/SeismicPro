import numpy as np
import pandas as pd

from .utils import to_list, get_cols, create_indexer


class SamplesContainer:
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
    @property
    def indexed_by(self):
        """str or list of str: Names of header indices."""
        index_names = list(self.headers.index.names)
        if len(index_names) == 1:
            return index_names[0]
        return index_names

    @property
    def n_traces(self):
        """int: The number of traces."""
        return len(self.headers)

    def __getitem__(self, key):
        """Select values of headers by their names.

        Notes
        -----
        A 2d array is always returned even for a single header.

        Parameters
        ----------
        key : str, list of str
            Names of headers to get values for.

        Returns
        -------
        result : 2d np.ndarray
            Headers values.
        """
        keys_array = np.array(to_list(key))
        if keys_array.dtype.type != np.str_:
            raise ValueError("Passed keys must be either str or array-like of str")
        return get_cols(self.headers, keys_array)

    def __setitem__(self, key, value):
        """Set given values to selected headers.

        Parameters
        ----------
        key : str or list of str
            Headers to set values for.
        value : np.ndarray
            Headers values to set.
        """
        key = to_list(key)
        val = pd.DataFrame(value, columns=key, index=self.headers.index)
        self.headers[key] = val


class GatherContainer(TraceContainer):
    def __len__(self):
        """The number of gathers."""
        return self.n_gathers

    @property
    def headers(self):
        """pd.DataFrame: loaded trace headers."""
        return self._headers

    @headers.setter
    def headers(self, headers):
        """Reconstruct an indexer on each headers assignment."""
        if not (headers.index.is_monotonic_increasing or headers.index.is_monotonic_decreasing):
            headers = headers.sort_index(kind="stable")
        self.indexer = create_indexer(headers.index)
        self._headers = headers

    @property
    def indices(self):
        """pd.Index: indices of gathers."""
        return self.indexer.unique_indices

    @property
    def n_gathers(self):
        """int: The number of gathers."""
        return len(self.indices)

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
        headers_indices = self.indexer.get_loc(indices)
        return self.headers.iloc[headers_indices]
