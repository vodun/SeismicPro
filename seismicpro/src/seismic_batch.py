"""Seismic batch.""" # pylint: disable=too-many-lines
import os
from itertools import product

import warnings
from textwrap import dedent
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
from sklearn.linear_model import LinearRegression
import pywt
import segyio

from ..batchflow import action, inbatch_parallel, Batch, any_action_failed

from .seismic_index import SegyFilesIndex, FieldIndex, KNNIndex, TraceIndex, CustomIndex

from .utils import (FILE_DEPENDEND_COLUMNS, partialmethod, calculate_sdc_for_field, massive_block,
                    check_unique_fieldrecord_across_surveys, calc_semblance, interpolate_velocities,
                    calc_bounds, calc_partial_semblance, calc_velocity_model)
from .file_utils import write_segy_file
from .plot_utils import IndexTracker, spectrum_plot, seismic_plot, statistics_plot, gain_plot, semblance_plot

INDEX_UID = 'TRACE_SEQUENCE_FILE'

PICKS_FILE_HEADER = 'FIRST_BREAK_TIME'
GEOM_CHECK_HEADER = 'CORRECT_GEOM'


ACTIONS_DICT = {
    "clip": (np.clip, "numpy.clip", "clip values"),
    "gradient": (np.gradient, "numpy.gradient", "gradient"),
    "fft2": (np.fft.fft2, "numpy.fft.fft2", "a Discrete 2D Fourier Transform"),
    "ifft2": (np.fft.ifft2, "numpy.fft.ifft2", "an inverse Discrete 2D Fourier Transform"),
    "fft": (np.fft.fft, "numpy.fft.fft", "a Discrete Fourier Transform"),
    "ifft": (np.fft.ifft, "numpy.fft.ifft", "an inverse Discrete Fourier Transform"),
    "rfft": (np.fft.rfft, "numpy.fft.rfft", "a real-input Discrete Fourier Transform"),
    "irfft": (np.fft.irfft, "numpy.fft.irfft", "a real-input inverse Discrete Fourier Transform"),
    "dwt": (pywt.dwt, "pywt.dwt", "a single level Discrete Wavelet Transform"),
    "idwt": (lambda x, *args, **kwargs: pywt.idwt(*x, *args, **kwargs), "pywt.idwt",
             "a single level inverse Discrete Wavelet Transform"),
    "wavedec": (pywt.wavedec, "pywt.wavedec", "a multilevel 1D Discrete Wavelet Transform"),
    "waverec": (lambda x, *args, **kwargs: pywt.waverec(list(x), *args, **kwargs), "pywt.waverec",
                "a multilevel 1D Inverse Discrete Wavelet Transform"),
    "pdwt": (lambda x, part, *args, **kwargs: pywt.downcoef(part, x, *args, **kwargs), "pywt.downcoef",
             "a partial Discrete Wavelet Transform data decomposition"),
    "cwt": (lambda x, *args, **kwargs: pywt.cwt(x, *args, **kwargs)[0].T, "pywt.cwt", "a Continuous Wavelet Transform"),
}


TEMPLATE_DOCSTRING = """
    Compute {description} for each trace.
    This method simply wraps ``apply_along_axis`` method by setting the
    ``func`` argument to ``{full_name}``.

    Parameters
    ----------
    src : str, optional
        Batch component to get the data from.
    dst : str, optional
        Batch component to put the result in.
    args : misc
        Any additional positional arguments to ``{full_name}``.
    kwargs : misc
        Any additional named arguments to ``{full_name}``.

    Returns
    -------
    batch : SeismicBatch
        Transformed batch. Changes ``dst`` component.
"""
TEMPLATE_DOCSTRING = dedent(TEMPLATE_DOCSTRING).strip()


def apply_to_each_component(method):
    """Combine list of src items and list dst items into pairs of src and dst items
    and apply the method to each pair.

    Parameters
    ----------
    method : callable
        Method to be decorated.

    Returns
    -------
    decorator : callable
        Decorated method.
    """
    def decorator(self, *args, src, dst=None, **kwargs):
        """Returned decorator."""
        if isinstance(src, str):
            src = (src, )
        if dst is None:
            dst = src
        elif isinstance(dst, str):
            dst = (dst, )

        res = []
        for isrc, idst in zip(src, dst):
            res.append(method(self, *args, src=isrc, dst=idst, **kwargs))
        return self if isinstance(res[0], SeismicBatch) else res
    return decorator

def add_actions(actions_dict, template_docstring):
    """Add new actions in ``SeismicBatch`` by setting ``func`` argument in
    ``SeismicBatch.apply_to_each_trace`` method to given callables.

    Parameters
    ----------
    actions_dict : dict
        A dictionary, containing new methods' names as keys and a callable,
        its full name and description for each method as values.
    template_docstring : str
        A string, that will be formatted for each new method from
        ``actions_dict`` using ``full_name`` and ``description`` parameters
        and assigned to its ``__doc__`` attribute.

    Returns
    -------
    decorator : callable
        Class decorator.
    """
    def decorator(cls):
        """Returned decorator."""
        for method_name, (func, full_name, description) in actions_dict.items():
            docstring = template_docstring.format(full_name=full_name, description=description)
            method = partialmethod(cls.apply_along_axis, func)
            method.__doc__ = docstring
            setattr(cls, method_name, method)

        return cls
    return decorator


@add_actions(ACTIONS_DICT, TEMPLATE_DOCSTRING)  # pylint: disable=too-many-public-methods,too-many-instance-attributes
class SeismicBatch(Batch):
    """Batch class for seimsic data. Contains seismic traces, metadata and processing methods.

    Parameters
    ----------
    index : TraceIndex
        Unique identifiers for sets of seismic traces.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to ``None``.

    Attributes
    ----------
    index : TraceIndex
        Unique identifiers for sets of seismic traces.
    meta : dict
        Metadata about batch components.
    components : tuple
        Array containing all component's name. Updated only by ``_init_component`` function
        if new component comes from ``dst`` or by ``load`` function.

    Note
    ----
    There are only two ways to add a new components to ``components`` attribute.
    1. Using parameter ``components`` in ``load``.
    2. Using parameter ``dst`` with init function named ``_init_component``.
    """
    def __init__(self, index, *args, preloaded=None, **kwargs):
        super().__init__(index, *args, preloaded=preloaded, **kwargs)
        if preloaded is None:
            self.meta = dict()

    #-------------------------------------------------------------------------#
    #                      Decorators & Support functions                     #
    #-------------------------------------------------------------------------#

    def _init_component(self, *args, dst, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args, kwargs
        dst = (dst, ) if isinstance(dst, str) else dst

        for comp in dst:
            self.meta[comp] = self.meta[comp] if comp in self.meta else dict()

            if self.components is None or comp not in self.components:
                self.add_components(comp, init=self.array_of_nones)
        return self.indices

    def _post_filter_by_mask(self, mask, *args, **kwargs):
        """Index filtration using all received masks. This post function assumes that
        components have already been filtered.

        Parameters
        ----------
        mask : list
            list of boolean arrays

        Returns
        -------
            : SeismicBatch
            New batch with new index.

        Note
        ----
        1. Batch items in each component should be filtered in decorated action.
        2. This post function creates new instance of SeismicBatch with new index
        instance and copies filtered components from original batch for elements
        in new index.
        """
        _ = args, kwargs
        if any_action_failed(mask):
            all_errors = [error for error in mask if isinstance(error, Exception)]
            print(all_errors)
            raise ValueError(all_errors)

        mask = np.concatenate(mask)
        new_idf = self.index.get_df(index=mask, reset=False)
        new_index = new_idf.index.unique()

        batch_index = type(self.index).from_index(index=new_index, idf=new_idf,
                                                  index_name=self.index.name)

        new_batch = type(self)(batch_index)
        new_batch.add_components(self.components, len(self.components) * [new_batch.array_of_nones])
        new_batch.meta = self.meta

        for isrc in new_batch.components:
            pos_new = new_batch.get_pos(None, isrc, new_batch.indices)
            pos_old = self.get_pos(None, isrc, new_batch.indices)
            getattr(new_batch, isrc)[pos_new] = getattr(self, isrc)[pos_old]
        return new_batch

    def copy_meta(self, from_comp, to_comp):
        """Copy meta from one component to another or from list of components to list of
        components with same length.

        Parameters
        ----------
        from_comp : str or array-like
            Component's name to copy meta from or list with names of components.
        to_comp : str or array-like
            Component's name to copy meta in or list with names of components.

        Raises
        ------
            ValueError : if `from_comp` and `to_comp` have different length.
            ValueError : if one of given to `from_comp` component doesn't exist.

        Returns
        -------
        batch : SeismicBatch
            Batch with new meta, components' data remains unchanged.

        Note
        ----
        If a component from `to_comp` has meta data, it will always be replaced with meta from
        the corresponding `from_comp`.
        """
        from_comp = (from_comp, ) if isinstance(from_comp, str) else from_comp
        to_comp = (to_comp, ) if isinstance(to_comp, str) else to_comp

        if len(from_comp) != len(to_comp):
            raise ValueError("Unexpected length of component's lists. Given len(from_comp)="
                             "{} != len(to_comp)={}.".format(len(to_comp), len(from_comp)))

        for fr_comp, t_comp in zip(from_comp, to_comp):
            if fr_comp not in self.meta:
                raise ValueError('{} does not exist.'.format(fr_comp))

            if fr_comp == t_comp:
                continue

            if self.meta[t_comp]:
                warnings.warn("Meta of component {} is not empty and".format(t_comp) + \
                              " will be replaced by the meta from component {}.".format(fr_comp),
                              UserWarning)
            self.meta[t_comp] = self.meta[fr_comp].copy()
        return self

    def items_viewer(self, src, scroll_step=1, **kwargs):
        """Scroll and view batch items. Emaple of use:
        ```
        %matplotlib notebook

        fig, tracker = batch.items_viewer('raw', vmin=-cv, vmax=cv, cmap='gray')
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
        ```

        Parameters
        ----------
        src : str
            The batch component with data to show.
        scroll_step : int, default: 1
            Number of batch items scrolled at one time.
        kwargs: dict
            Additional keyword arguments for plt.

        Returns
        -------
        fig, tracker
        """
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, getattr(self, src), self.indices,
                               scroll_step=scroll_step, **kwargs)
        return fig, tracker

    #-------------------------------------------------------------------------#
    #                              Load and Dump                              #
    #-------------------------------------------------------------------------#

    @action
    def load(self, src=None, fmt=None, components=None, **kwargs):
        """Load data into components.

        Parameters
        ----------
        src : misc, optional
            Source to load components from.
        fmt : str, optional
            Source format.
        components : str or array-like, optional
            Components to load.
        **kwargs: dict
            Any kwargs to be passed to load method.

        Returns
        -------
        batch : SeismicBatch
            Batch with loaded components.
        """
        if fmt.lower() in ['sgy', 'segy']:
            return self._load_segy(src=components, dst=components, **kwargs)
        if fmt == 'picks':
            return self._load_from_index(src=PICKS_FILE_HEADER, dst=components)
        if fmt == 'index':
            return self._load_from_index(src=src, dst=components)
        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    @apply_to_each_component
    def _load_segy(self, src, dst, tslice=None):
        """Load data from segy files.

        Parameters
        ----------
        src : str, array-like
            Component to load.
        dst : str, array-like
            The batch component to put loaded data in.
        tslice: slice, optional
            Load a trace subset given by slice.

        Returns
        -------
        batch : SeismicBatch
            Batch with loaded components.
        """
        segy_index = SegyFilesIndex(self.index, name=src)
        sdf = segy_index.get_df()
        sdf['order'] = np.arange(len(sdf))
        order = self.index.get_df().merge(sdf)['order']

        batch = type(self)(segy_index)._load_from_segy_file(src=src, dst=dst, tslice=tslice) # pylint: disable=protected-access
        all_traces = np.concatenate(getattr(batch, dst))[order]
        self.meta[dst] = batch.meta[dst]

        if self.index.name is None:
            res = np.array(list(np.expand_dims(all_traces, 1)) + [None])[:-1]
        else:
            lens = self.index.tracecounts
            res = np.array(np.split(all_traces, np.cumsum(lens)[:-1]) + [None])[:-1]

        self.add_components(dst, init=res)
        return self

    @inbatch_parallel(init="_init_component", target="threads")
    def _load_from_segy_file(self, index, *args, src, dst, tslice=None):
        """Load from a single segy file."""
        _ = src, args
        pos = self.get_pos(None, "indices", index)
        path = index
        trace_seq = self.index.get_df([index])[(INDEX_UID, src)]
        if tslice is None:
            tslice = slice(None)

        # Infering cube geometry may be time consuming for some `segy` files.
        # Set `ignore_geometry = True` to skip this stage when opening `segy` file.
        with segyio.open(path, strict=False, ignore_geometry=True) as segyfile:
            traces = np.atleast_2d([segyfile.trace[i - 1][tslice] for i in
                                    np.atleast_1d(trace_seq).astype(int)])
            samples = segyfile.samples[tslice]
            interval = segyfile.bin[segyio.BinField.Interval]

        getattr(self, dst)[pos] = traces
        if index == self.indices[0]:
            self.meta[dst]['samples'] = samples
            self.meta[dst]['interval'] = interval
            self.meta[dst]['sorting'] = None
        return self

    @apply_to_each_component
    def _load_from_index(self, src, dst):
        """Load picking from dataframe column."""
        idf = self.index.get_df(reset=False)
        ind = np.cumsum(self.index.tracecounts)[:-1]
        dst_data = np.split(idf[src].values, ind)
        self.add_components(dst, init=np.array(dst_data + [None])[:-1])
        self.meta.update({dst:dict(sorting=None)})
        return self

    @action
    def dump(self, src, fmt, path, **kwargs):
        """Export data to file.

        Parameters
        ----------
        src : str
            Batch component to dump data from.
        fmt : str
            Output data format.

        Returns
        -------
        batch : SeismicBatch
            Unchanged batch.
        """
        if fmt.lower() in ['sgy', 'segy']:
            return self._dump_segy(src, path, **kwargs)
        if fmt == 'picks':
            return self._dump_picking(src, path, **kwargs)
        if fmt == 'geom':
            return self._dump_geometry_flags(src, path, **kwargs)
        raise NotImplementedError('Unknown format.')

    def _dump_segy(self, src, path, split=True):
        """Dump data to segy files.

        Parameters
        ----------
        path : str
            Path for output files.
        src : str
            Batch component to dump data from.
        split : bool
            Whether to dump batch items into separate files.

        Returns
        -------
        batch : SeismicBatch
            Unchanged batch.
        """
        if split:
            return self._dump_split_segy(src, path)

        return self._dump_single_segy(src, path)

    @inbatch_parallel(init="indices", target="threads")
    def _dump_split_segy(self, index, src, path):
        """Dump data to segy files."""
        pos = self.get_pos(None, src, index)
        data = np.atleast_2d(getattr(self, src)[pos])

        path = os.path.join(path, str(index) + '.sgy')

        df = self.index.get_df([index], reset=False)
        sort_by = self.meta[src]['sorting']
        if sort_by is not None:
            df = df.sort_values(by=sort_by)

        df.reset_index(drop=self.index.name is None, inplace=True)
        headers = list(set(df.columns.levels[0]) - set(FILE_DEPENDEND_COLUMNS))
        segy_headers = [h for h in headers if hasattr(segyio.TraceField, h)]
        df = df[segy_headers]
        df.columns = df.columns.droplevel(1)

        write_segy_file(data, df, self.meta[src]['samples'], path)
        return self

    def _dump_single_segy(self, src, path):
        """Dump data to segy file."""
        data = np.vstack(getattr(self, src))

        df = self.index.get_df(reset=False)
        sort_by = self.meta[src]['sorting']
        if sort_by is not None:
            df = df.sort_values(by=sort_by)

        df = df.loc[self.indices]
        df.reset_index(drop=self.index.name is None, inplace=True)
        headers = list(set(df.columns.levels[0]) - set(FILE_DEPENDEND_COLUMNS))
        segy_headers = [h for h in headers if hasattr(segyio.TraceField, h)]
        df = df[segy_headers]
        df.columns = df.columns.droplevel(1)

        write_segy_file(data, df, self.meta[src]['samples'], path)
        return self

    @action
    def _dump_picking(self, src, path, src_traces, input_units='samples', columns=('FieldRecord', 'TraceNumber')):
        """Dump picking to file.

        Parameters
        ----------
        src : str
            Source to get picking from.
        path : str
            Output file path.
        src_traces : str
            Batch component with corresponding traces.
        input_units : str
            Units in which picking is stored in src. Must be one of the 'samples' or 'milliseconds'.
            In case 'milliseconds' dumped as is. Otherwise converted to milliseconds first.
        columns: array_like
            Columns to include in the output file.
            In case `PICKS_FILE_HEADER` not included it will be added automatically.

        Returns
        -------
        batch : SeismicBatch
            Batch unchanged.
        """
        if not isinstance(self.index, TraceIndex):
            raise ValueError('Picking dump works with TraceIndex only')
        data = getattr(self, src)
        if input_units == 'samples':
            data = data.astype(int)
            data = self.meta[src_traces]['samples'][data]

        df = self.index.get_df()[list(columns)]
        df.columns = df.columns.droplevel(1)

        df[PICKS_FILE_HEADER] = data

        if not os.path.isfile(path):
            df.to_csv(path, index=False, header=True, mode='a')
        else:
            df.to_csv(path, index=False, header=None, mode='a')
        return self

    @action
    def _dump_geometry_flags(self, src, path, columns=('FieldRecord',)):
        """Dump results of check for geometry assignment correctness to file.

        Parameters
        ----------
        src : str
            Source to get flags from.
        path : str
            Output file path.
        columns: array_like
            Columns to include in the output file.
            In case `CORRECT_GEOM` not included it will be added automatically.

        Returns
        -------
        batch : SeismicBatch
            Batch unchanged.
        """
        if not isinstance(self.index, FieldIndex):
            raise ValueError('Geometry check dump works with FieldIndex only')
        data = getattr(self, src)

        df = self.index.get_df(reset=True)[list(columns)].drop_duplicates()
        df.columns = df.columns.droplevel(1)

        df[GEOM_CHECK_HEADER] = data

        if not os.path.isfile(path):
            df.to_csv(path, index=False, header=True, mode='a')
        else:
            df.to_csv(path, index=False, header=None, mode='a')
        return self

    #-------------------------------------------------------------------------#
    #                           Data Process Actions                          #
    #-------------------------------------------------------------------------#

    #-------------------------------------------------------------------------#
    #                          DPA. Cropping Actions                          #
    #-------------------------------------------------------------------------#

    @action
    @inbatch_parallel(init='_init_component')
    @apply_to_each_component
    def make_grid_for_crops(self, index, src, dst, shape, drop_last=True):
        """ Generate coordinates for crops that cover all seismogram

        Parameters
        ----------
        src : str
            component from which the crops will be cropped
        dst : str
            component to store crops coordinates
        shape : tuple of ints
            crop shape
        drop_last: bool
            If True, drop border crops if they are incomplete
        """

        if isinstance(self.index, SegyFilesIndex):
            raise NotImplementedError("Index can't be SegyFilesIndex")

        pos = self.get_pos(None, None, index)
        field = getattr(self, src)[pos]

        len_x, len_y = field.shape
        x, y = shape

        coords_x = np.arange(0, len_x, x)
        if len_x % x != 0 and drop_last:
            coords_x = coords_x[:-1]

        coords_y = np.arange(0, len_y, y)
        if len_y % y != 0 and drop_last:
            coords_y = coords_y[:-1]

        getattr(self, dst)[pos] = list(product(coords_x, coords_y))
        return self

    @action
    @apply_to_each_component
    def crop(self, src, coords, shape, dst=None, pad_zeros=False):
        """ Crop from seismograms by given coordinates.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        coords: list, NamedExpression
            The list with tuples (x,y) of top-left coordinates for each crop.
                - if `coords` is the list then crops from the same coords for each item in the batch.
                - if `coords` is an `R` NamedExpression it should return values in [0, 1) with shape
                  (num_crops, 2). Same coords will be sampled for each item in the batch.
                - if `coords` is the list of lists wrapped in `P` NamedExpression and len(coords) equals
                  to batch size, then crops from individual coords for each item in the batch.
                - if `coords` is `P(R(..))` NamedExpression, `R` should return values in [0, 1) with shape
                  (num_crops, 2) and different coords will be sampled for each batch item.
        shape: tuple of ints
            Crop shape.
        pad_zeros: bool
            Wether to zero-pad incomplete crops. Valid only for absolute coordinates

        Returns
        -------
            : SeismicBatch
            Batch with crops. `dst` components are now arrays (of size batch items) of arrays (number of crops)
            of arrays (crop shape).

        Raises
        ------
        ValueError : if shape is larger than seismogram in any dimension.
        ValueError : if coord + shape is larger than seismogram in any dimension.

        Notes
        -----
        1. Works properly only with FieldIndex.
        2. `R` samples a relative position of top-left coordinate in a feasible region of seismogram.

        Examples
        --------

        ::

            crop(src=['raw', 'mask], dst=['raw_crop', 'mask_crop], coords=[[0, 0], [1, 1]], shape=(100, 256))
            crop(src=['raw', 'mask], dst=['raw_crop', 'mask_crop], shape=(100, 256),
                coords=P([[[0, 0]], [[0, 0], [2, 2]]])).next_batch(2)
            crop(src=['raw', 'mask], dst=['raw_crop', 'mask_crop], shape=(100, 256),
                coords=P(R('uniform', size=(N_RANDOM_CROPS, 2)))).next_batch(2)
        """
        if isinstance(self.index, SegyFilesIndex):
            raise NotImplementedError("Index can't be SegyFilesIndex")

        self._init_component(dst=dst)
        self.meta[dst]['crop_coords'] = {}
        self.meta[dst]['crops_source'] = src

        self.__crop(src, coords, shape, pad_zeros, dst)
        return self

    @inbatch_parallel(init='indices')
    def __crop(self, index, src, coords, shape, pad_zeros, dst=None):
        """ Generate crops from an array with seismic data
        see :meth:`~SeismicBatch.crop` for full description
        """
        pos = self.get_pos(None, src, index)
        arr = getattr(self, src)[pos]

        if all(((0 <= x < 1) and (0 <= y < 1)) for x, y in coords):
            feasible_region = np.array(arr.shape) - shape
            xy = (feasible_region * coords).astype(int)
            if np.any(xy < 0):
                raise ValueError("`shape` is larger than one of seismogram's dimensions")
        else:
            xy = np.array(coords)

        res = np.empty((len(xy), *shape))
        for i, (x, y) in enumerate(xy):
            if pad_zeros:
                crop = np.zeros(shape)

                x1 = min(x + shape[0], arr.shape[0])
                y1 = min(y + shape[1], arr.shape[1])

                crop[:x1-x, :y1-y] = arr[x:x1, y:y1]
                res[i] = crop

            else:
                if (x + shape[0] > arr.shape[0]) or (y + shape[1] > arr.shape[1]):
                    raise ValueError('Coordinates', (x, y), 'exceed feasible region of seismogram with shape',
                                     arr.shape, ', with crop shape', shape, 'but pad_zeros is False')
                res[i] = arr[x:x+shape[0], y:y+shape[1]]

        getattr(self, dst)[pos] = res

        self.meta[dst]['crop_coords'][index] = xy

    @action
    @inbatch_parallel(init='_init_component')
    @apply_to_each_component
    def assemble_crops(self, index, src, dst, fill_value=0.0):
        """
        Assembles crops from `src` into a single seismogram

        Parameters
        ----------
        src : str
            component with crops
        dst : str
            component to put the result to.
        fill_value : float
            the area that is not covered with crops is filled with this value
        """

        if isinstance(self.index, SegyFilesIndex):
            raise NotImplementedError("Index can't be SegyFilesIndex")

        if 'crop_coords' not in self.meta[src]:
            raise ValueError("{} component doesn't contain crops!".format(src))

        pos = self.get_pos(None, None, index)
        crops = getattr(self, src)[pos]
        coords = self.meta[src]['crop_coords'][index]

        res_x = self.index.tracecounts[pos]
        res_y = len(self.meta[self.meta[src]['crops_source']]['samples'])

        res = np.full((res_x, res_y), fill_value, dtype=float)
        crop_counts = np.zeros((res_x, res_y))

        for crop_coords, crop in zip(coords, crops):
            x, y = crop_coords
            len_x, len_y = crop.shape

            x1 = min(x+len_x, res_x)
            y1 = min(y+len_y, res_y)

            res[x:x1, y:y1] = crop[:x1-x, :y1-y]
            crop_counts[x:x1, y:y1] += 1

        crop_counts[crop_counts == 0] = 1

        getattr(self, dst)[pos] = res / crop_counts
        return self

    #-------------------------------------------------------------------------#
    #                          DPA. Normalize actions                         #
    #-------------------------------------------------------------------------#

    @action
    def standardize(self, src, dst):
        """Standardize traces to zero mean and unit variance.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.

        Returns
        -------
        batch : SeismicBatch
            Batch with the standardized traces.

        Note
        ----
        This action copies all meta from `src` component to `dst` component.
        """
        data = np.concatenate(getattr(self, src))
        std_data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 10 ** -6)

        traces_in_item = [len(i) for i in getattr(self, src)]
        ind = np.cumsum(traces_in_item)[:-1]

        dst_data = np.split(std_data, ind)
        setattr(self, dst, np.array(dst_data + [None])[:-1]) # array implicitly converted to object dtype
        self.copy_meta(src, dst)
        return self

    @action
    @inbatch_parallel(init='_init_component')
    @apply_to_each_component
    def equalize(self, index, src, dst, params, survey_id_col=None, upscale=False):
        """ Equalize amplitudes of different seismic surveys in dataset.

        This method performs quantile normalization by shifting and
        scaling data in each batch item so that 95% of absolute values
        seismic surveys that item belongs to lie between 0 and 1.

        `params` argument should contain a dictionary in a following form:

        {survey_name: 95th_perc, ...},

        where `95_perc` is an estimate for 95th percentile of absolute
        values for seismic survey with `survey_name`.

        One way to obtain such a dictionary is to use
        `SeismicDataset.find_equalization_params' method, which calculates
        esimated and saves them to `SeismicDataset`'s attribute. This method
        can be used from pipeline.

        Other way is to provide user-defined dictionary for `params` argument.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.
        params : dict or NamedExpr
            Containter with parameters for equalization.
        survey_id_col : str, optional
            Column in index that indicate names of seismic
            surveys from different seasons.
            Optional if `params` is a result of `SeismicDataset`'s
            method `find_equalization_params`.
        upscale : bool, optional
            weather to upscale batch items to its origin scale

        Returns
        -------
            : SeismicBatch
            Batch of shot gathers with equalized data.

        Raises
        ------
        ValueError : If gather with same id is contained in more
                     than one survey.

        Note
        ----
        1. If `params` dict is user-defined, `survey_id_col` should be
        provided excplicitly either as argument, or as `params` dict key-value
        pair.
        2. This action copies all meta from `src` component to `dst` component.
        """
        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]

        if survey_id_col is None:
            survey_id_col = params['survey_id_col']

        surveys_by_fieldrecord = np.unique(self.index.get_df(index=index, reset=False)[survey_id_col])
        check_unique_fieldrecord_across_surveys(surveys_by_fieldrecord, index)
        survey = surveys_by_fieldrecord[0]

        p_95 = params[survey]

        # shifting and scaling data so that 5th and 95th percentiles are -1 and 1 respectively
        equalized_field = (field / p_95) if not upscale else (field * p_95)

        getattr(self, dst)[pos] = equalized_field
        self.copy_meta(src, dst)
        return self

    #-------------------------------------------------------------------------#
    #                        DPA. velocity Law Actions                        #
    #-------------------------------------------------------------------------#

    @action
    def calculate_semblance(self, src, dst, velocities, velocities_length=100, window=25):
        r"""
        TODO: THIS DOCSTRING SHOULD BE REWRITTEN!

        Calculate semblance for given fields from `src` component and save resulted semblance to `dst` component.
        Semblance is a measure of multichannel coherence. This measure is calculated along all possible horizons from
        a given velocity range and with all possible starting points. Range of starting points is equal to field range.
        Calculation of each horizon is based on the following formula.
        :math:`th_i = \sqrt{t_z^2 + off_i^2/v^2}`, where
        :math:`t_z` - start time of the horizon
        :math:`off_i` - distance from the gather to the i-th trace (offset)
        :math:`v` - velocity for this horizon
        :math:`th_i` - horizon time for given offset.
        The value of horison amplitude in point t_h can be received from trace with same offset. In order to
        calculate all the horizon values, you need to get the t_h values for all offsets.
        Then, value :math:`f_{i, th_i} = trace_i[th_i]` is an amplitude for horizon with i-th offset.
        Semblance calculate base on horizon values by the following formula:
        :math:`S(k, th) = \frac{\sum^{k+N/2}_{k-N/2}(\sum^M_{i=1} \sum^M_{j=1} f_{i, th[j]})^2}
                               {M \sum^{k+N/2}_{k-N/2}\sum^M_1 \sum^M_{j=1} f_{i, th[j]})^2}`, where
        S - semblance value for range for starting point k
        M - Number of traces in field
        N - Window size
        th - Horizon.
        Resulted matrix contains semblance values based on horizons with each combinations of started point :math:`t_z`
        and velocity :math:`v`. This matrix has shape (time_length, velocity_length).
        Parameters
        ----------
        src : str
            The batch component to get field from.
        dst : str
            The batch component to put semblance in.
        velocities : list with length 2 or more.
            If length is 2, then the checked velocities will be sampled with given step via `step` parameter
            from first value to the second one. If length is different, semblance will be calculated for given
            velocities.
        step: int
            Step to sample velocity.
        window: int
            Window size for smoothing. It should be from 5 to 256ms.
        Returns
        -------
            : SeismicBatch
            Batch with semblance in `dst` component. `dst` components are now arrays
            (of size batch items) of array (of size velocity values) of array (of field length).
        Raises
        ------
        ValueError : if `method` receive wrong name.
        Notes
        -----
        - Works properly only with CDP index.
        - Adding 'velocity' to meta data.
        """

        if len(velocities) == 2:
            velocities = np.linspace(*velocities, velocities_length)

        # Converting ms to seconds.
        times = self.meta[src]['samples'].astype(int) / 1000
        dt = times[1] - times[0]

        self._calculate_semblance(src=src, dst=dst, times=times, velocities=velocities, dt=dt,
                                   window=window)
        self.copy_meta(src, dst)
        self.meta[dst].update(velocities=velocities)
        return self

    @action
    @inbatch_parallel(init="_init_component", target="for")
    def _calculate_semblance(self, index, src, dst, times, velocities, dt, window):
        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]
        offsets = np.sort(self.index.get_df(index=index)['offset'])

        field = np.ascontiguousarray(field.T)
        semblance = calc_semblance(field, times, offsets, velocities, dt, window)

        getattr(self, dst)[pos] = semblance

    @action
    def calculate_residual_semblance(self, src, dst, velocities, velocity_points, velocities_length=100, window=25, p=0.2):
        """ some docs """
        if len(velocities) == 2:
            velocities = np.linspace(*velocities, velocities_length)
        times = self.meta[src]['samples'].astype(int) / 1000
        dt = times[1] - times[0]

        self._calc_residual_semblance(src=src, dst=dst, times=times, velocities=velocities, velocity_points=velocity_points,
                                      dt=dt, window=window, p=p)

        return self

    @inbatch_parallel(init="_init_component", target="for")
    def _calc_residual_semblance(self, index, src, dst, times, velocities, velocity_points, dt, window, p):
        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]
        offsets = np.sort(self.index.get_df(index=index)['offset'])

        interp_vel = interpolate_velocities(velocity_points, times)
        lower_bounds, upper_bounds = calc_bounds(interp_vel, velocities, p)
        semblance = calc_partial_semblance(seismogram=field.T, times=times, offsets=offsets, velocities=velocities,
                                           lower_bounds=lower_bounds, upper_bounds=upper_bounds, dt=dt, window=window)
        semblance_len = (upper_bounds - lower_bounds).max()
        residual_semblance = np.zeros((len(times), semblance_len))

        for i, (low, up) in enumerate(zip(lower_bounds, upper_bounds)):
            ix = (semblance_len - (up - low)) // 2
            residual_semblance[i, ix : ix + up - low] = semblance[i, low : up]

        getattr(self, dst)[pos] = residual_semblance

    @action
    def find_velocity_model(self, src, dst, area_factor=0.1):
        """ some docs """
        times = self.meta[src]['samples'].astype(int) / 1000
        velocities = self.meta[src].get('velocities', None)

        if velocities is None:
            raise ValueError('Src should contains semblance.')

        self._find_velocity_model(src=src, dst=dst, times=times, velocities=velocities, area_factor=area_factor)
        return self

    @inbatch_parallel(init='_init_component', target='threads')
    def _find_velocity_model(self, index, src, dst, times, velocities, area_factor):
        pos = self.get_pos(None, src, index)
        semblance = getattr(self, src)[pos]
        semblance = np.ascontiguousarray(semblance)

        velocity_points, velocity_metrics = calc_velocity_model(semblance=semblance, times=times,
                                                          velocities=velocities, area_factor=area_factor)
        velocity_points[:, 0] *= 1000 # from m / ms to m / s
        self.copy_meta(src, dst)
        self.meta[dst].update(velocity_metrics=velocity_metrics)
        getattr(self, dst)[pos] = velocity_points

    @action
    @inbatch_parallel(init='_init_component', target='threads')
    @apply_to_each_component
    def add_muting(self, index, src, dst, interp_type='polyfit', muting=None, picking=None, indent=None):
        """muting - two arrays. first - offsets, second - time.
        Picking is a component with picking values for every field in batch.
        indet is a list with length of 2 or list with length = len(field).
        """
        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]
        offsets = np.sort(self.index.get_df(index=index)['offset'])
        t_step = np.diff(self.meta[src]['samples'][:2])[0]
        data_x, data_y = None, None
        indent = indent if indent is not None else 0

        if picking is not None:
            picking = getattr(self, picking)[pos]
            data_x = picking[:len(picking)].copy() # ms
            data_y = offsets[:len(picking)].copy() # meters
        elif muting is not None:
            indent = np.zeros(len(field))
            data_x, data_y = muting
        else:
            raise ValueError('Either `picking` or `muting` should be determined.')

        if interp_type == 'polyfit':
            poly = np.polyfit(data_x, data_y, deg=2)
            mute_samples = np.polyval(poly, offset) / t_step
            if np.sum(mute_samples < 0) > 0:
                mute_samples[mute_samples < 0] = 0
        elif interp_type == 'regression':
            lin_reg = LinearRegression(fit_intercept=False)
            lin_reg.fit(data_x.reshape(-1, 1), data_y)
            indent /= 1000 # to ms
            velocity = lin_reg.coef_ - indent # reduce velocity by given indent. lin_reg.coef_ has m / ms
            mute_samples = data_y / (velocity * t_step) # meters / samples
            mute_samples = mute_samples.astype(int)
        else:
            raise ValueError('Interpolation type must be "ployfit" or "regression" not "{}"'.format(interp_type))
        mute_time_mx = np.array([mute_samples]*field.shape[1]).T

        time_idx = np.arange(0, field.shape[1])
        time_idx_mx = np.array([time_idx]*field.shape[0])

        mute_mask = time_idx_mx - mute_time_mx
        mute_mask[mute_mask < 0] = 0
        mute_mask = mute_mask.astype(bool)
        muted_field = field * mute_mask
        getattr(self, dst)[pos] = muted_field
        self.copy_meta(src, dst)

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    def get_coords_fm_supergather(self, index, src, dst):
        """docs"""
        pos = self.index.get_pos(index)
        coords_x, coords_y = self.index.get_df(index=index)[src].values.T

        coords_x, coords_y = np.unique(coords_x), np.unique(coords_y)

        coords_x = coords_x[len(coords_x) // 2]
        coords_y = coords_y[len(coords_y) // 2]
        getattr(self, dst)[pos] = np.array([coords_x, coords_y])

    #-------------------------------------------------------------------------#
    #                                DPA. Misc                                #
    #-------------------------------------------------------------------------#

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def apply_along_axis(self, index, func, *args, src, dst=None, slice_axis=0, **kwargs):
        """Apply function along specified axis of batch items.

        Parameters
        ----------
        func : callable
            A function to apply. Must accept a trace as its first argument.
        src : str, array-like
            Batch component name to get the data from.
        dst : str, array-like
            Batch component name to put the result in.
        item_axis : int, default: 0
            Batch item axis to apply ``func`` along.
        slice_axis : int
            Axis to iterate data over.
        args : misc
            Any additional positional arguments to ``func``.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        batch : SeismicBatch
            Transformed batch. Changes ``dst`` component.
        """
        i = self.get_pos(None, src, index)
        src_data = getattr(self, src)[i]
        dst_data = np.array([func(x, *args, **kwargs) for x in np.rollaxis(src_data, slice_axis)])
        getattr(self, dst)[i] = dst_data

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def band_pass_filter(self, index, *args, src, dst=None, lowcut=None, highcut=None, fs=1, order=5):
        """Apply a band pass filter.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        lowcut : real, optional
            Lowcut frequency.
        highcut : real, optional
            Highcut frequency.
        order : int
            The order of the filter.
        fs : real
            Sampling rate.

        Returns
        -------
        batch : SeismicBatch
            Batch with filtered traces.
        """
        _ = args
        i = self.get_pos(None, src, index)
        traces = getattr(self, src)[i]
        nyq = 0.5 * fs
        if lowcut is None:
            b, a = signal.butter(order, highcut / nyq, btype='high')
        elif highcut is None:
            b, a = signal.butter(order, lowcut / nyq, btype='low')
        else:
            b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')

        getattr(self, dst)[i] = signal.lfilter(b, a, traces)

    @action
    def correct_spherical_divergence(self, src, dst, speed, params, time=None):
        """Correction of spherical divergence with given parameers or with optimal parameters.

        There are two ways to use this funcion. The simplest way is to determine parameters then
        correction will be made with given parameters. Another approach is to find the parameters
        by ```find_sdc_params``` function from `SeismicDataset` class. In this case, optimal
        parameters can be stored in in dataset's attribute or pipeline variable and then passed
        to this action as `params` argument.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.
        speed : array
            Wave propagation speed depending on the depth.
            Speed is measured in milliseconds.
        params : array of floats(or ints) with length 2
            Containter with parameters in the following order: [v_pow, t_pow].
        time : array, optional
            Trace time values. If `None` defaults to self.meta[src]['samples'].
            Time measured in either in samples or in milliseconds.

        Returns
        -------
            : SeismicBatch
            Batch of shot gathers with corrected spherical divergence.

        Raises
        ------
        ValueError : If Index is not FieldIndex.
        ValueError : If length of ```params``` not equal to 2.

        Note
        ----
        1. Works properly only with FieldIndex.
        2. This action copies all meta from `src` component to `dst` component.
        """
        if not isinstance(self.index, FieldIndex):
            raise ValueError("Index must be FieldIndex, not {}".format(type(self.index)))

        if len(params) != 2:
            raise ValueError("The length of the ```params``` must be equal to two, not {}.".format(len(params)))

        time = self.meta[src]['samples'] if time is None else np.array(time, dtype=int)
        step = np.diff(time[:2])[0].astype(int)
        speed = np.array(speed, dtype=int)[::step]
        v_pow, t_pow = params

        self._correct_sph_div(src=src, dst=dst, time=time, speed=speed, v_pow=v_pow, t_pow=t_pow)
        self.copy_meta(src, dst)
        return self

    @inbatch_parallel(init='_init_component')
    def _correct_sph_div(self, index, src, dst, time, speed, v_pow, t_pow):
        """Correct spherical divergence with given parameters. """
        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]

        correct_field = calculate_sdc_for_field(field, time, speed, v_pow=v_pow, t_pow=t_pow)

        getattr(self, dst)[pos] = correct_field
        return self

    @action
    @inbatch_parallel(init="indices", post='_post_filter_by_mask', target="threads")
    def drop_zero_traces(self, index, src, num_zero, all_comps_sorted=True):
        """Drop traces with sequence of zeros longer than ```num_zero```.

        This action drops traces from index dataframe and from all batch components
        according to the mask calculated on `src` component.

        Parameters
        ----------
        num_zero : int
            All traces that contain more than `num_zero` consecutive zeros will be removed.
        src : str, array-like
            The batch components to get the data from.
        all_comps_sorted : bool
            Check that all components have the same sorting to ensure that they are
            modified in a same way.

        Returns
        -------
            : SeismicBatch
            Batch without dropped traces.

        Raises
        ------
        ValueError : if `src` has no sorting and batch index is FieldIndex.
        ValueError : if `all_comps_sorted` is True and any component in batch has
                     sorting different from `src`.

        Note
        ----
        This action creates new instance of SeismicBatch with new index
        instance.
        """
        sorting = self.meta[src]['sorting']
        if sorting is None and not isinstance(self.index, TraceIndex):
            raise ValueError('traces in `{}` component should be sorted '
                             'before dropping zero traces'.format(src))

        if all_comps_sorted:
            has_same_sorting = all(self.meta[comp]['sorting'] == sorting for comp in self.components)
            if not has_same_sorting:
                raise ValueError('all components in batch should have same sorting')

        pos = self.get_pos(None, src, index)
        traces = getattr(self, src)[pos]
        mask = list()
        for trace in traces:
            nonzero_indices = np.flatnonzero(trace)
            # add -1 and len(trace) indices to count leading and trailing zero sequences
            nonzero_indices = np.concatenate(([-1], nonzero_indices, [len(trace)]))
            zero_seqs = np.diff(nonzero_indices) - 1
            mask.append(np.max(zero_seqs) < num_zero)
        mask = np.array(mask)

        for comp in self.components:
            getattr(self, comp)[pos] = getattr(self, comp)[pos][mask]

        if sorting:
            cols = [(INDEX_UID, src), (sorting, '')]
            index_df = self.index.get_df([index])
            if cols[0] not in index_df.columns:
                # Level 1 of MultiIndex contains name of common columns ('') at first position and names of
                # all columns that relate to specific sgy files.
                raise ValueError('`src` should be one of the component names that Index was created with: {}'
                                 ''.format(index_df.columns.levels[1][1:].values))
            sorted_index_df = index_df[cols].sort_values(sorting)
            order = np.argsort(sorted_index_df[cols[0]].values, kind='stable')
            return mask[order]
        return mask

    @action
    @inbatch_parallel(init='_init_component')
    @apply_to_each_component
    def hodograph_straightening(self, index, velocities, src=None, dst=None, num_mean_tr=0):
        r""" Straightening up the travel time curve with normal grading.
        Shifted time is calculated as follows:

        $$ t_new = \sqrt{t_0^2 + l^2 / V^2} $$

        If ```num_mean_tr``` can be evaluated to True,
        new amplitude value for t_0 is the mean value of ```num_mean_tr```'s adjacent amplitudes from t_new.

        Parameters
        ----------
        velocities : 1-d array or 2-d array
            Speed law for traces.
            If 1-d array of same length as traces - array of velocities(m/s) in each time stamp
            If 2-d array - it is interpreted as array of pairs (time(ms), velocity(m/s))
            and velocities in each time stamp are interpolated. Time should increase.
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        num_mean_tr : int or None, optional
            Number of timestamps for smoothing new amplitude value. If 0 (default) or None, no smoothing is performed

        Returns
        -------
            : SeismicBatch
            Traces straightened on the basis of speed and time values.

        Raises
        ------
        ValueError : Raise if traces are not sorted by offset.

        Note
        ----
        1. Works only with sorted traces by offset.
        2. Works properly only with CustomIndex with CDP index.
        3. This action copies all meta from `src` component to `dst` component.
        """
        if not isinstance(self.index, CustomIndex):
            raise ValueError("Index must be CustomIndex, not {}".format(type(self.index)))

        index_name = self.index.get_df(reset=False).index.name
        if index_name != 'CDP':
            raise ValueError("Index name must be CDP, not {}".format(index_name))

        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]

        offset = np.sort(self.index.get_df(index=index)['offset'])

        if self.meta[src]['sorting'] != 'offset':
            raise ValueError('All traces should be sorted by offset not {}'.format(self.meta[src]['sorting']))
        if 'samples' in self.meta[src].keys():
            time_range_ms = self.meta[src]['samples']
            sample_time = time_range_ms[1] - time_range_ms[0]
            num_timestamps = len(time_range_ms)
        else:
            raise ValueError('`sample_time` should be present in `self.meta[{}]`'.format(src))

        velocities = np.array(velocities)
        if velocities.ndim == 2 and velocities.shape[1] == 2:
            if not np.all(np.diff(velocities[:, 0]) > 0):
                raise ValueError('Sample velocities times are not increasing!')
            speed_conc = np.interp(time_range_ms, velocities[:, 0], velocities[:, 1])
        elif velocities.ndim == 1 and velocities.shape[0] == num_timestamps:
            speed_conc = velocities
        else:
            raise ValueError('Velocities specified incorrectly!')

        speed_conc /= 1000  # convert from m/s to m/ms

        mean_traces = None
        if num_mean_tr:
            left = -int(num_mean_tr/2) + (~num_mean_tr % 2)
            right = left + num_mean_tr
            mean_traces = np.arange(left, right).reshape(-1, 1)

        new_field = []

        for ix, off in enumerate(offset):
            new_time_ms = np.sqrt(time_range_ms**2 + (off/speed_conc)**2)
            new_ts = np.round(new_time_ms / sample_time)

            if mean_traces is not None:
                ix_to_mean = np.stack([new_ts]*num_mean_tr) + mean_traces
                ix_to_mean = np.clip(ix_to_mean, 0, num_timestamps - 1).astype(int)

                new_field.append(np.mean(field[ix][ix_to_mean], axis=0))
            else:
                new_ts = np.clip(new_ts, 0, num_timestamps - 1).astype(int)
                new_field.append(field[ix][new_ts])

        getattr(self, dst)[pos] = np.array(new_field)
        self.copy_meta(src, dst)
        return self

    @action
    def mcm(self, src, dst, eps=3, length_win=12):
        """Creates for each trace corresponding Energy function.
        Based on Coppens(1985) method.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.
        eps: float, default: 3
            Stabilization constant that helps reduce the rapid fluctuations of energy function.
        length_win: int, default: 12
            The leading window length.

        Returns
        -------
        batch : SeismicBatch
            Batch with the energy function.
        """
        trace = np.concatenate(getattr(self, src))
        energy = np.cumsum(trace**2, axis=1)
        long_win, lead_win = energy, energy
        lead_win[:, length_win:] = lead_win[:, length_win:] - lead_win[:, :-length_win]
        energy = lead_win / (long_win + eps)
        self.add_components(dst, init=np.array(energy + [None])[:-1]) # array implicitly converted to object dtype
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def pad_traces(self, index, *args, src, dst=None, **kwargs):
        """
        Pad traces with ```numpy.pad```.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        kwargs : dict
            Named arguments to ```numpy.pad```.

        Returns
        -------
        batch : SeismicBatch
            Batch with padded traces.

        Note
        ----
        This action copies all meta from `src` component to `dst` component.
        """
        _ = args
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        pad_width = kwargs['pad_width']
        if isinstance(pad_width, int):
            pad_width = (pad_width, pad_width)

        kwargs['pad_width'] = [(0, 0)] + [pad_width] + [(0, 0)] * (data.ndim - 2)
        getattr(self, dst)[pos] = np.pad(data, **kwargs)
        self.copy_meta(src, dst)
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def slice_traces(self, index, *args, src, slice_obj, dst=None):
        """
        Slice traces.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        slice_obj : slice
            Slice to extract from traces.

        Returns
        -------
        batch : SeismicBatch
            Batch with sliced traces.

        Note
        ----
        This action copies all meta from `src` component to `dst` component.
        """
        _ = args
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        getattr(self, dst)[pos] = data[:, slice_obj]
        self.copy_meta(src, dst)
        return self

    @action
    @apply_to_each_component
    def sort_traces(self, *args, src, sort_by, dst=None):
        """Sort traces.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        sort_by : str
            Sorting key.

        Returns
        -------
        batch : SeismicBatch
            Batch with new trace sorting.

        Note
        ----
        1. This action copies all meta from `src` component to `dst` component.
        2. `dst` meta contains sorting type in `meta['sorting']`.
        """
        _ = args
        if src in self.meta.keys():
            current_sorting = self.meta[src].get('sorting')
        else:
            current_sorting = None

        if current_sorting == sort_by:
            return self

        self._sort(src=src, sort_by=sort_by, current_sorting=current_sorting, dst=dst)
        self.copy_meta(src, dst)
        self.meta[dst]['sorting'] = sort_by
        return self

    @inbatch_parallel(init="_init_component", target="threads")
    def _sort(self, index, src, sort_by, current_sorting, dst=None):
        """Sort traces.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        sort_by : str
            Sorting key.
        current_sorting : str
            Current sorting of `src` component

        Returns
        -------
        batch : SeismicBatch
            Batch with new trace sorting.
        """
        pos = self.get_pos(None, src, index)
        df = self.index.get_df([index])

        if current_sorting:
            cols = [current_sorting, sort_by]
            sorted_index_df = df[cols].sort_values(current_sorting)
            order = np.argsort(sorted_index_df[sort_by].values, kind='stable')
        else:
            order = np.argsort(df[sort_by].values, kind='stable')

        getattr(self, dst)[pos] = getattr(self, src)[pos][order]
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def to_2d(self, index, *args, src, dst=None, length_alignment=None, pad_value=0):
        """Convert array of 1d arrays to 2d array.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        length_alignment : str, optional
            Defines what to do with arrays of diffetent lengths.
            If 'min', cut the end by minimal array length.
            If 'max', pad the end to maximal array length.
            If None, try to put array to 2d array as is.

        Returns
        -------
        batch : SeismicBatch
            Batch with items converted to 2d arrays.
        """
        _ = args
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        if data is None or len(data) == 0:
            return

        try:
            data_2d = np.vstack(data)
        except ValueError as err:
            if length_alignment is None:
                raise ValueError('Try to set length_alingment to \'max\' or \'min\'') from err
            if length_alignment == 'min':
                nsamples = min([len(t) for t in data])
            elif length_alignment == 'max':
                nsamples = max([len(t) for t in data])
            else:
                raise NotImplementedError('Unknown length_alingment') from err
            shape = (len(data), nsamples)
            data_2d = np.full(shape, pad_value)
            for i, arr in enumerate(data):
                data_2d[i, :len(arr)] = arr[:nsamples]

        getattr(self, dst)[pos] = data_2d

    def trace_headers(self, header, flatten=False):
        """Get trace heades.

        Parameters
        ----------
        header : string
            Header name.
        flatten : bool
            If False, array of headers will be splitted according to batch item sizes.
            If True, return a flattened array. Dafault to False.

        Returns
        -------
        arr : ndarray
            Arrays of trace headers."""
        tracecounts = self.index.tracecounts
        values = self.index.get_df()[header].values
        if flatten:
            return values
        return np.array(np.split(values, np.cumsum(tracecounts)[:-1]) + [None])[:-1]

    #-------------------------------------------------------------------------#
    #                           DPA. Picking Actions                          #
    #-------------------------------------------------------------------------#

    @action
    def picking_to_mask(self, src, dst, src_traces):
        """Convert picking time to the mask for TraceIndex.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.
        src_traces : str
            The batch components which contains traces.

        Returns
        -------
        batch : SeismicBatch
            Batch with the mask corresponds to the picking.
        """
        data = np.concatenate(getattr(self, src))

        samples = self.meta[src_traces]['samples']
        rate = samples[1] - samples[0]
        data = np.around(data / rate).astype('int')

        batch_size = data.shape[0]
        trace_length = getattr(self, src_traces)[0].shape[1]
        ind = tuple(np.array(list(zip(range(batch_size), data))).T)
        ind[1][ind[1] < 0] = 0
        mask = np.zeros((batch_size, trace_length))
        mask[ind] = 1
        dst_data = np.cumsum(mask, axis=1)

        traces_in_item = [len(i) for i in getattr(self, src)]
        ind = np.cumsum(traces_in_item)[:-1]

        dst_data = np.split(dst_data, ind)
        dst_data = np.array([np.squeeze(i) for i in dst_data] + [None])[:-1]
        setattr(self, dst, dst_data)
        return self

    @action
    def mask_to_pick(self, src, dst, labels=True):
        """Convert the mask to picking time. Piciking time corresponds to the
        begininning of the longest block of consecutive ones in the mask.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.
        labels: bool, default: False
            The flag indicates whether action's inputs probabilities or labels.

        Returns
        -------
        batch : SeismicBatch
            Batch with the predicted picking times.
        """
        data = getattr(self, src)
        if not labels:
            data = np.argmax(data, axis=1)

        dst_data = massive_block(data)
        setattr(self, dst, np.array(dst_data + [None])[:-1]) # array implicitly converted to object dtype
        return self

    @inbatch_parallel(init='_init_component', target="threads")
    def shift_pick_phase(self, index, src, src_traces, dst=None, shift=1.5, threshold=0.05):
        """ Shifts picking time stored in `src` component on the given phase along the traces stored in `src_traces`.

        Parameters
        ----------
        src : str
            The batch component to get picking from.
        dst : str
            The batch component to put the result in.
        src_traces: str
            The batch component where the traces are stored.
        shift: float
            The amount of phase to shift measured in radians. Default is 1.5 , which corresponds
            to transfering the picking times from 'max' to 'zero' type.
        threshold: float
            Threshold determining amplitude, such that all the samples with amplitude less then threshold would be
            skipped. Introduced because of unstable behaviour of the hilbert transform at the begining of the signal.
         """
        shift *= np.pi
        pos = self.get_pos(None, src, index)
        pick = getattr(self, src)[pos]
        trace = getattr(self, src_traces)[pos]
        if isinstance(self.index, KNNIndex):
            trace = trace[0]
        trace = np.squeeze(trace)

        analytic = hilbert(trace)
        phase = np.unwrap(np.angle(analytic))
        # finding x such that phase[x] = phase[pick] - shift
        phase_mod = phase - (phase[pick] - shift)
        phase_mod[phase_mod < 0] = 0
        # in case phase_mod reaches 0 multiple times find the index of last one
        x = len(phase_mod) - phase_mod[::-1].argmin() - 1
        # skip the trace samples with amplitudes < threshold, starting from the `zero` sample
        n_skip = max((np.abs(trace[x:]) > threshold).argmax() - 1, 0)
        x += n_skip
        getattr(self, dst)[pos] = x
        return self

    @action
    def energy_to_picking(self, src, dst):
        """Convert energy function of the trace to the picking time by taking derivative
        and finding maximum.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.

        Returns
        -------
        batch : SeismicBatch
            Batch with the predicted picking by MCM method.
        """
        energy = np.stack(getattr(self, src))
        energy = np.gradient(energy, axis=1)
        picking = np.argmax(energy, axis=1)
        self.add_components(dst, np.array(picking + [None])[:-1]) # array implicitly converted to object dtype
        return self

    #-------------------------------------------------------------------------#
    #                                 Plotters                                #
    #-------------------------------------------------------------------------#

    def seismic_plot(self, src, index, wiggle=False, xlim=None, ylim=None, std=1, # pylint: disable=too-many-arguments
                     src_picking=None, s=None, scatter_color=None, figsize=None,
                     save_to=None, dpi=None, line_color=None, title=None, **kwargs):
        """Plot seismic traces.

        Parameters
        ----------
        src : str or array of str
            The batch component(s) with data to show.
        index : same type as batch.indices
            Data index to show.
        wiggle : bool, default to False
            Show traces in a wiggle form.
        xlim : tuple, optionalgit
            Range in x-axis to show.
        ylim : tuple, optional
            Range in y-axis to show.
        std : scalar, optional
            Amplitude scale for traces in wiggle form.
        src_picking : str
            Component with picking data.
        s : scalar or array_like, shape (n, ), optional
            The marker size in points**2.
        scatter_color : color, sequence, or sequence of color, optional
            The marker color.
        figsize : array-like, optional
            Output plot size.
        save_to : str or None, optional
            If not None, save plot to given path.
        dpi : int, optional, default: None
            The resolution argument for matplotlib.pyplot.savefig.
        line_color : color, sequence, or sequence of color, optional, default: None
            The trace color.
        title : str
            Plot title.
        kwargs : dict
            Additional keyword arguments for plot.

        Returns
        -------
        Multi-column subplots.
        """
        pos = self.get_pos(None, 'indices', index)
        if len(np.atleast_1d(src)) == 1:
            src = (src,)

        if src_picking is not None:
            rate = self.meta[src[0]]['interval'] / 1e3
            picking = getattr(self, src_picking)[pos] / rate
            pts_picking = (range(len(picking)), picking)
        else:
            pts_picking = None

        arrs = [getattr(self, isrc)[pos] for isrc in src]
        names = [' '.join([i, str(index)]) for i in src]
        seismic_plot(arrs=arrs, wiggle=wiggle, xlim=xlim, ylim=ylim, std=std,
                     pts=pts_picking, s=s, scatter_color=scatter_color,
                     figsize=figsize, names=names, save_to=save_to,
                     dpi=dpi, line_color=line_color, title=title, **kwargs)
        return self

    def crops_plot(self, src, index, # pylint: disable=too-many-arguments
                   num_crops=None,
                   wiggle=False, std=1,
                   src_picking=None, s=None, scatter_color=None,
                   figsize=None, title=None,
                   save_to=None, dpi=None, **kwargs):
        """Plot seismic traces.

        Parameters
        ----------
        src : str or array of str
            The batch component(s) with crops to show.
        index : same type as batch.indices
            Data index to show.
        num_crops: int or None
            If not None, random `num_crops` crops are shown, all crops otherwise
        wiggle : bool, default to False
            Show traces in a wiggle form.
        std : scalar, optional
            Amplitude scale for traces in wiggle form.
        src_picking : str
            Component with picking data.
        s : scalar or array_like, shape (n, ), optional
            The marker size in points**2.
        scatter_color : color, sequence, or sequence of color, optional
            The marker color.
        figsize : array-like, optional
            Output plot size.
        save_to : str or None, optional
            If not None, save plot to given path.
        dpi : int, optional, default: None
            The resolution argument for matplotlib.pyplot.savefig.
        title : str
            Plot title.
        kwargs : dict
            Additional keyword arguments for plot.

        Returns
        -------
        Multi-column subplots.
        """

        if 'crop_coords' not in self.meta[src]:
            raise ValueError("{} component doesn't contain crops!".format(src))

        pos = self.get_pos(None, 'indices', index)

        if src_picking is not None:
            raise NotImplementedError()

        pts_picking = None

        arrs = getattr(self, src)[pos]
        total_crops = len(arrs)
        names = self.meta[src]['crop_coords'][index]

        if num_crops is not None and num_crops < len(arrs):
            crops_indices = np.random.choice(np.arange(len(arrs)), size=num_crops, replace=False)
            arrs = arrs[crops_indices]
            names = names[crops_indices]
        else:
            num_crops = len(arrs)

        names = [str(c) for c in names]
        title = "{} (of {}) crops from {}".format(num_crops, total_crops, index)

        seismic_plot(arrs=arrs, wiggle=wiggle, std=std,
                     pts=pts_picking, s=s, scatter_color=scatter_color,
                     figsize=figsize, names=names, save_to=save_to,
                     dpi=dpi, title=title, **kwargs)
        return self

    def gain_plot(self, src, index, window=51, xlim=None, ylim=None,
                  figsize=None, names=None, **kwargs):
        """Gain's graph plots the ratio of the maximum mean value of
        the amplitude to the mean value of the amplitude at the moment t.

        Parameters
        ----------
        window : int, default 51
            Size of smoothing window of the median filter.
        xlim : tuple or list with size 2
            Bounds for plot's x-axis.
        ylim : tuple or list with size 2
            Bounds for plot's y-axis.
        figsize : array-like, optional
            Output plot size.
        names : str or array-like, optional
            Title names to identify subplots.

        Returns
        -------
        Gain's plot.
        """
        _ = kwargs
        pos = self.get_pos(None, 'indices', index)
        src = (src, ) if isinstance(src, str) else src
        sample = [getattr(self, source)[pos] for source in src]
        gain_plot(sample, window, xlim, ylim, figsize, names, **kwargs)
        return self

    def spectrum_plot(self, src, index, frame, max_freq=None,
                      figsize=None, save_to=None, **kwargs):
        """Plot seismogram(s) and power spectrum of given region in the seismogram(s).

        Parameters
        ----------
        src : str or array of str
            The batch component(s) with data to show.
        index : same type as batch.indices
            Data index to show.
        frame : tuple
            List of slices that frame region of interest.
        max_freq : scalar
            Upper frequence limit.
        figsize : array-like, optional
            Output plot size.
        save_to : str or None, optional
            If not None, save plot to given path.
        kwargs : dict
            Named argumets to matplotlib.pyplot.imshow.

        Returns
        -------
        Plot of seismogram(s) and power spectrum(s).
        """
        pos = self.get_pos(None, 'indices', index)
        if len(np.atleast_1d(src)) == 1:
            src = (src,)

        arrs = [getattr(self, isrc)[pos] for isrc in src]
        names = [' '.join([i, str(index)]) for i in src]
        rate = self.meta[src[0]]['interval'] / 1e6
        spectrum_plot(arrs=arrs, frame=frame, rate=rate, max_freq=max_freq,
                      names=names, figsize=figsize, save_to=save_to, **kwargs)
        return self

    def statistics_plot(self, src, index, stats, figsize=None, save_to=None, **kwargs):
        """Plot seismogram(s) and various trace statistics.

        Parameters
        ----------
        src : str or array of str
            The batch component(s) with data to show.
        index : same type as batch.indices
            Data index to show.
        stats : str, callable or array-like
            Name of statistics in statistics zoo, custom function to be avaluated or array of stats.
        figsize : array-like, optional
            Output plot size.
        save_to : str or None, optional
            If not None, save plot to given path.
        kwargs : dict
            Named argumets to matplotlib.pyplot.imshow.

        Returns
        -------
        Plot of seismogram(s) and power spectrum(s).
        """
        pos = self.get_pos(None, 'indices', index)
        if len(np.atleast_1d(src)) == 1:
            src = (src,)

        arrs = [getattr(self, isrc)[pos] for isrc in src]
        names = [' '.join([i, str(index)]) for i in src]
        rate = self.meta[src[0]]['interval'] / 1e6
        statistics_plot(arrs=arrs, stats=stats, rate=rate, names=names, figsize=figsize,
                        save_to=save_to, **kwargs)
        return self

    def semblance_plot(self, src, index, x_ticks=30, x_step=None, y_ticks=30, y_step=None, samples_step=None,
                       velocity_points=None, point_size=50, name=None, figsize=None, save_dir=None, dpi=100, **kwargs):
        """ Semblance plot.
        TODO: REWRITE DOCS
        Parameters
        ----------
         src : str
            The batch component to get semblance from.
        index : same type as batch.indices
            Data index to show.
        figsize : tuple with length 2
            size of output graph.
        Raises
        ------
        ValueError : if semblance isn't calculated.
        """
        velocities = self.meta[src].get('velocities')
        if velocities is None:
            raise ValueError(f'There is no semblance in {src} variable.')
        pos = self.get_pos(None, src, index)
        semblance = getattr(self, src)[pos]

        samples = self.meta[src].get('samples')
        samples_step = samples_step if samples is None else samples[1] - samples[0]

        semblance_plot(semblance=semblance, velocities=velocities, x_ticks=x_ticks, x_step=x_step,
                       y_ticks=y_ticks, y_step=y_step, samples_step=samples_step, velocity_points=velocity_points,
                       point_size=point_size, name=name, index=index, figsize=figsize, save_dir=save_dir, dpi=dpi)
        return self
