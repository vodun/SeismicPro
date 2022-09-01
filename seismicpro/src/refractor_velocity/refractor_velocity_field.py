"""Implements a RefractorVelocityField class which stores near-surface velocity models calculated at different field
location and allows for their spatial interpolation"""

from textwrap import dedent
from functools import partial, cached_property

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .refractor_velocity import RefractorVelocity
from .interactive_plot import FitPlot
from .utils import get_param_names, postprocess_params, calc_df_to_dump, load_rv, dump_rv
from ..field import SpatialField
from ..utils import to_list, Coordinates, IDWInterpolator


class RefractorVelocityField(SpatialField):
    """A class for storing near-surface velocity models calculated at different field location and interpolating them
    spatially over the whole field.

    Refractor velocities used to compute a depth model of the very first layers are usually estimated on a sparse grid
    of inlines and crosslines and then interpolated over the whole field in order to reduce computational costs. Such
    interpolation can be performed by `RefractorVelocityField` which provides an interface to obtain a velocity model
    of an upper part of the section at given spatial coordinates via its `__call__` and `interpolate` methods.

    A field can be populated with refractor velocities in 2 main ways:
    - by passing precalculated velocities in the `__init__`,
    - by creating an empty field and then iteratively updating it with estimated velocities using `update`.

    After all velocities are added, field interpolator should be created to make the field callable. It can be done
    either manually by executing `create_interpolator` method or automatically during the first call to the field if
    `auto_create_interpolator` flag was set to `True` upon field instantiation. Manual interpolator creation is useful
    when one wants to fine-tune its parameters or the field should be later passed to different processes (e.g. in a
    pipeline with prefetch with `mpc` target) since otherwise the interpolator will be independently created in all the
    processes.

    Examples
    --------
    A field can be created empty and updated with instances of `RefractorVelocity` class:
    >>> field = RefractorVelocityField()
    >>> rv = RefractorVelocity(t0=100, x1=1500, v1=1600, v2=2200,
                               coords=Coordinates((150, 80), names=("INLINE_3D", "CROSSLINE_3D")))
    >>> field.update(rv)

    Or created from precalculated instances:
    >>> field = RefractorVelocityField(list_of_rv)

    Note that in both these cases all velocity models in the filed must describe the same number of refractors.

    Velocity models of an upper part of the section are usually estimated independently of one another and thus may
    appear inconsistent. `refine` method allows utilizing local information about near-surface conditions to refit
    the field:
    >>> field = field.refine()

    Only fields that were constructed directly from offset-traveltime data can be refined.

    Field interpolator will be created automatically upon the first call by default, but one may do it explicitly by
    executing `create_interpolator` method:
    >>> field.create_interpolator("rbf")

    Now the field allows for velocity interpolation at given coordinates:
    >>> rv = field((100, 100))

    Or can be passed directly to some gather processing methods:
    >>> gather = survey.sample_gather().apply_lmo(field)

    Parameters
    ----------
    items : RefractorVelocity or list of RefractorVelocity, optional
        Velocity models to be added to the field on instantiation. If not given, an empty field is created.
    survey : Survey, optional
        A survey described by the field.
    is_geographic : bool, optional
        Coordinate system of the field: either geographic (e.g. (CDP_X, CDP_Y)) or line-based (e.g. (INLINE_3D,
        CROSSLINE_3D)). Inferred automatically on the first update if not given.
    auto_create_interpolator : bool, optional, defaults to True
        Whether to automatically create default interpolator upon the first call to the field.

    Attributes
    ----------
    survey : Survey or None
        A survey described by the field. `None` if not specified during instantiation.
    item_container : dict
        A mapping from coordinates of field items as 2-element tuples to the items themselves.
    is_geographic : bool
        Whether coordinate system of the field is geographic. `None` for an empty field if was not specified during
        instantiation.
    coords_cols : tuple with 2 elements or None
        Names of SEG-Y trace headers representing coordinates of items in the field. `None` if names of coordinates are
        mixed or the field is empty.
    interpolator : SpatialInterpolator or None
        Field data interpolator.
    is_dirty_interpolator : bool
        Whether the field was updated after the interpolator was created.
    auto_create_interpolator : bool
        Whether to automatically create default interpolator upon the first call to the field.
    """
    item_class = RefractorVelocity

    def __init__(self, items=None, n_refractors=None, survey=None, is_geographic=None, auto_create_interpolator=True):
        self.n_refractors = n_refractors
        super().__init__(items, survey, is_geographic, auto_create_interpolator)

    

    @property
    def param_names(self):
        """list of str: Names of model parameters."""
        if self.n_refractors is None:
            raise ValueError("The number of refractors is undefined")
        return get_param_names(self.n_refractors)

    @cached_property
    def is_fit(self):
        """bool: Whether the field was constructed directly from offset-traveltime data."""
        return all(item.is_fit for item in self.items)

    @cached_property
    def max_offset(self):
        """float: Mean maximum offset reliably described by field items."""
        max_offsets = [item.max_offset for item in self.items if item.max_offset is not None]
        if max_offsets:
            return np.mean(max_offsets)
        return None

    @cached_property
    def mean_velocity(self):
        """RefractorVelocity: Mean near-surface velocity model over the field."""
        return self.construct_item(self.values.mean(axis=0), coords=None)

    def __str__(self):
        """Print field metadata including descriptive statistics of the near-surface velocity model, coordinate system
        and created interpolator."""
        msg = super().__str__() + dedent(f"""\n
        Number of refractors:      {self.n_refractors}
        Mean max offset of items:  {self.max_offset}
        Is fit from first breaks:  {self.is_fit}
        """)

        if not self.is_empty:
            params_df = pd.DataFrame(self.values, columns=self.param_names)
            params_stats_str = params_df.describe().iloc[1:].T.to_string(col_space=8, float_format="{:.02f}".format)
            msg += f"""\nDescriptive statistics of the near-surface velocity model:\n{params_stats_str}"""

        return msg

    def validate_items(self, items):
        """Check if the field can be updated with the provided `items`."""
        super().validate_items(items)
        n_refractors_set = {item.n_refractors for item in items}
        if self.n_refractors is not None:
            n_refractors_set.add(self.n_refractors)
        if len(n_refractors_set) != 1:
            raise ValueError("Each RefractorVelocity must describe the same number of refractors as the field")

    @classmethod
    def from_file(cls, path, is_geographic=None, encoding="UTF-8"):
        """Load RefractorVelocityField from a file.

        File should have coords and parameters of a single RefractorVelocity with next structure:
         - The first row contain the Coordinates parameters names (name_x, name_y, coord_x, coord_y) and
        the RefractorVelocity parameters names ("t0", "x1"..."x{n-1}", "v1"..."v{n}", "max_offset").
         - Each next line contains row contains the coords names, coords values, and parameters values of one
        RefractorVelocity.

        File example:
         name_x     name_y    coord_x    coord_y        t0        x1        v1        v2 max_offset
        SourceX    SourceY    1111100    2222220     50.00   1000.00   1500.00   2000.00    2000.00
        ...
        SourceX    SourceY    1111200    2222240     60.00   1050.00   1550.00   1950.00    2050.00

        Parameters
        ----------
        path : str,
            path to the file.
        encoding : str, defaults to "UTF-8"
            File encoding.

        Returns
        -------
        self : RefractorVelocityField
            RefractorVelocityField instance created from a file.
        """
        coords_list, params_list, max_offset_list = load_rv(path, encoding)
        rv_list = []
        for coords, params, max_offset in zip(coords_list, params_list, max_offset_list):
            rv = RefractorVelocity(max_offset=max_offset, coords=coords, **params)
            rv_list.append(rv)
        return cls(rv_list, is_geographic=is_geographic)

    def update(self, items):
        """Add new items to the field. All passed `items` must have not-None coordinates and describe the same number
        of refractors as the field.

        Parameters
        ----------
        items : RefractorVelocity or list of RefractorVelocity
            Items to add to the field.

        Returns
        -------
        self : RefractorVelocityField
            `self` with new items added. Changes `item_container` inplace and sets the `is_dirty_interpolator` flag to
            `True` if the `items` list is not empty. Sets `is_geographic` flag and `n_refractors` attribute during the
            first update if they were not defined during field creation. Resets `coords_cols` attribute if headers,
            defining coordinates of any item being added, differ from those of the field.

        Raises
        ------
        TypeError
            If wrong type of items were found.
        ValueError
            If any of the passed items have `None` coordinates or describe not the same number of refractors as the
            field.
        """
        items = to_list(items)
        super().update(items)
        if items:
            self.n_refractors = items[0].n_refractors
        return self

    @staticmethod
    def item_to_values(item):
        """Convert a field item to a 1d `np.ndarray` of its values being interpolated."""
        return np.array(list(item.params.values()))

    def _interpolate(self, coords):
        """Interpolate field values at given `coords` and postprocess them so that the following constraints are
        satisfied:
        - Intercept time is non-negative,
        - Crossover offsets are non-negative and increasing,
        - Velocities of refractors are non-negative and increasing.
        `coords` are guaranteed to be a 2d `np.ndarray` with shape (n_coords, 2), converted to the coordinate system of
        the field."""
        values = self.interpolator(coords)
        return postprocess_params(values)

    def construct_item(self, values, coords):
        """Construct an instance of `RefractorVelocity` from its `values` at given `coords`."""
        return self.item_class(**dict(zip(self.param_names, values)), max_offset=self.max_offset, coords=coords)

    def _get_refined_values(self, interpolator_class, min_refractor_points=0, min_refractor_points_quantile=0):
        """Redefine parameters of velocity models for refractors that contain small number of points and may thus have
        produced noisy estimates during fitting.

        Parameters of such refractors are redefined using an interpolator of the given type constructed over all
        well-fit data of the field.

        The number of points in a refractor of a given field item is considered to be small if it is less than:
        - 2 or `min_refractor_points`,
        - A quantile of the number of points in the very same refractor over the whole field defined by
          `min_refractor_points_quantile`.
        """
        coords = self.coords
        values = self.values
        refined_values = np.empty_like(values)

        # Calculate the number of point in each refractor for velocity models that were fit
        n_refractor_points = np.full((self.n_items, self.n_refractors), fill_value=np.nan)
        for i, rv in enumerate(self.item_container.values()):
            if rv.is_fit:
                n_refractor_points[i] = np.histogram(rv.offsets, rv.piecewise_offsets, density=False)[0]
        n_refractor_points[:, np.isnan(n_refractor_points).all(axis=0)] = 0

        # Calculate minimum acceptable number of points in each refractor, should be at least 2
        min_refractor_points = np.maximum(np.nanquantile(n_refractor_points, min_refractor_points_quantile, axis=0),
                                          max(2, min_refractor_points))
        ignore_mask = n_refractor_points < min_refractor_points
        ignore_mask[:, ignore_mask.all(axis=0)] = False  # Use a refractor anyway if it is ignored for all items

        # Refine t0 using only items with well-fitted first refractor
        refined_values[:, 0] = interpolator_class(coords[~ignore_mask[:, 0]], values[~ignore_mask[:, 0], 0])(coords)

        # Refine crossover offsets using only items with well-fitted neighboring refractors
        for i in range(1, self.n_refractors):
            proper_items_mask = ~(ignore_mask[:, i - 1] | ignore_mask[:, i])
            refined_values[:, i] = interpolator_class(coords[proper_items_mask], values[proper_items_mask, i])(coords)

        # Refine velocities using only items with well-fitted corresponding refractor
        for i in range(self.n_refractors, 2 * self.n_refractors):
            proper_items_mask = ~ignore_mask[:, i - self.n_refractors]
            refined_values[:, i] = interpolator_class(coords[proper_items_mask], values[proper_items_mask, i])(coords)

        # Postprocess refined values
        return postprocess_params(refined_values)

    def create_interpolator(self, interpolator, min_refractor_points=0, min_refractor_points_quantile=0, **kwargs):
        """Create a field interpolator whose name is defined by `interpolator`.

        Available options are:
        - "idw" - to create `IDWInterpolator`,
        - "delaunay" - to create `DelaunayInterpolator`,
        - "ct" - to create `CloughTocherInterpolator`,
        - "rbf" - to create `RBFInterpolator`.

        Parameters
        ----------
        interpolator : str
            Name of the interpolator to create.
        min_refractor_points : int, optional, defaults to 0
            Ignore parameters of refractors with less than `min_refractor_points` points during interpolation.
        min_refractor_points_quantile : float, optional, defaults to 0
            Defines quantiles of the number of points in each refractor of the field. Parameters of refractors with
            less points than the corresponding quantile are ignored during interpolation.
        kwargs : misc, optional
            Additional keyword arguments to be passed to the constructor of interpolator class.

        Returns
        -------
        field : Field
            A field with created interpolator. Sets `is_dirty_interpolator` flag to `False`.
        """
        interpolator_class = self._get_interpolator_class(interpolator)
        values = self._get_refined_values(interpolator_class, min_refractor_points, min_refractor_points_quantile)
        self.interpolator = interpolator_class(self.coords, values, **kwargs)
        self.is_dirty_interpolator = False
        return self

    def smooth(self, radius=None, neighbors=4, min_refractor_points=0, min_refractor_points_quantile=0):
        """Smooth the field by averaging its velocity models within given radius.

        Parameters
        ----------
        radius : positive float, optional
            Spatial window radius (Euclidean distance). Equals to `self.default_neighborhood_radius` if not given.
        neighbors : int, optional, defaults to 4
            The number of neighbors to use for averaging if no velocities are considered to be well-fit in given
            `radius` according to provided `min_refractor_points` and `min_refractor_points_quantile`.
        min_refractor_points : int, optional, defaults to 0
            Ignore parameters of refractors with less than `min_refractor_points` points during averaging.
        min_refractor_points_quantile : float, optional, defaults to 0
            Defines quantiles of the number of points in each refractor of the field. Parameters of refractors with
            less points than the corresponding quantile are ignored during averaging.

        Returns
        -------
        field : RefractorVelocityField
            Smoothed field.
        """
        if self.is_empty:
            return type(self)(survey=self.survey, is_geographic=self.is_geographic)
        if radius is None:
            radius = self.default_neighborhood_radius
        smoother = partial(IDWInterpolator, radius=radius, neighbors=neighbors, dist_transform=0)
        values = self._get_refined_values(smoother, min_refractor_points, min_refractor_points_quantile)

        smoothed_items = []
        for rv, val in zip(self.items, values):
            item = self.construct_item(val, rv.coords)

            # Copy all fit-related items from the parent field
            item.is_fit = rv.is_fit
            item.fit_result = rv.fit_result
            item.init = rv.init
            item.bounds = rv.bounds
            item.offsets = rv.offsets
            item.times = rv.times

            smoothed_items.append(item)

        return type(self)(smoothed_items, n_refractors=self.n_refractors, survey=self.survey,
                          is_geographic=self.is_geographic)

    def dump(self, path, encoding="UTF-8", min_col_size=11):
        """Save the RefractorVelocityField instance to a file.

        The file must have the coordinates and parameters of a single RefractorVelocity with the following structure:
        The first line contains coords names and parameter names ("t0", "x1"..."x{n-1}", "v1"..."v{n}", "max_offset").
        Each next line contains the coords and parameters values corresponding to a single RefractorVelocity in the
        resulting RefractorVelocityField.

        File example:
        SourceX   SourceY        t0        x1        v1        v2 max_offset
        1111100   2222220     50.00   1000.00   1500.00   2000.00    2000.00
        ...
        1111200   2222240     60.00   1050.00   1550.00   1950.00    2050.00

        Parameters
        ----------
        path : str
            Path to the file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.
        min_col_size : int, defaults to 11
            Minimum size of each columns in the resulting file.

        Returns
        -------
        self : RefractorVelocityField
            RefractorVelocityField unchanged.

        Raises
        ------
        ValueError
            If RefractorVelocityField is empty.
        """
        if self.is_empty:
            raise ValueError("Field is empty. Could not dump empty field.")
        df_list = [calc_df_to_dump(rv) for rv in self.item_container.values()]
        dump_rv(df_list, path=path, encoding=encoding, min_col_size=min_col_size)
        return self

    def refine(self, radius=None, neighbors=4, min_refractor_points=0, min_refractor_points_quantile=0,
               relative_bounds_size=0.25, bar=True):
        """Refine the field by first smoothing it and then refitting each velocity model within narrow parameter bounds
        around smoothed values. Only fields that were constructed directly from offset-traveltime data can be refined.

        Parameters
        ----------
        radius : positive float, optional
            Spatial window radius for smoothing (Euclidean distance). Equals to `self.default_neighborhood_radius` if
            not given.
        neighbors : int, optional, defaults to 4
            The number of neighbors to use for smoothing if no velocities are considered to be well-fit in given
            `radius` according to provided `min_refractor_points` and `min_refractor_points_quantile`.
        min_refractor_points : int, optional, defaults to 0
            Ignore parameters of refractors with less than `min_refractor_points` points during smoothing.
        min_refractor_points_quantile : float, optional, defaults to 0
            Defines quantiles of the number of points in each refractor of the field. Parameters of refractors with
            less points than the corresponding quantile are ignored during smoothing.
        relative_bounds_size : float, optional, defaults to 0.25
            Size of parameters bound used to refit velocity models relative to their range in the smoothed field. The
            bounds are centered around smoothed parameter values.
        bar : bool, optional, defaults to True
            Whether to show a refinement progress bar.

        Returns
        -------
        field : RefractorVelocityField
            Refined field.
        """
        if not self.is_fit:
            raise ValueError("Only fields that were constructed directly from offset-traveltime data can be refined")
        smoothed_field = self.smooth(radius, neighbors, min_refractor_points, min_refractor_points_quantile)
        bounds_size = smoothed_field.values.ptp(axis=0) * relative_bounds_size / 2
        params_bounds = np.stack([smoothed_field.values - bounds_size, smoothed_field.values + bounds_size], axis=2)

        # Clip t0 bounds to be positive and all crossover bounds to be no greater than max offset
        max_offsets = np.array([rv.max_offset for rv in self.items])[:, None, None]
        params_bounds[:, 0] = np.maximum(params_bounds[:, 0], 0)
        params_bounds[:, 1:self.n_refractors] = np.minimum(params_bounds[:, 1:self.n_refractors], max_offsets)

        refined_items = []
        for rv, bounds in tqdm(zip(smoothed_field.items, params_bounds), total=self.n_items,
                               desc="Velocity models refined", disable=not bar):
            rv = RefractorVelocity.from_first_breaks(rv.offsets, rv.times, bounds=dict(zip(self.param_names, bounds)),
                                                     max_offset=rv.max_offset, coords=rv.coords)
            refined_items.append(rv)
        return type(self)(refined_items, n_refractors=self.n_refractors, survey=self.survey,
                          is_geographic=self.is_geographic)

    def dump(self, path, encoding="UTF-8", min_col_size=11):
        """Save the RefractorVelocityField instance to a file.

        The resulting file have the coordinates and parameters of a single RefractorVelocity with the following
        structure:
         - The first line contain the Coordinates parameters names (name_x, name_y, coord_x, coord_y) and
        the RefractorVelocity parameters names ("t0", "x1"..."x{n-1}", "v1"..."v{n}", "max_offset").
         - Each next line contains the coords names, coords values, and parameters values corresponding to one
        RefractorVelocity in the resulting RefractorVelocityField.

        File example:
         name_x     name_y    coord_x    coord_y        t0        x1        v1        v2 max_offset
        SourceX    SourceY    1111100    2222220     50.00   1000.00   1500.00   2000.00    2000.00
        ...
        SourceX    SourceY    1111200    2222240     60.00   1050.00   1550.00   1950.00    2050.00

        Parameters
        ----------
        path : str
            Path to the file.
        encoding : str, optional, defaults to "UTF-8"
            File encoding.
        min_col_size : int, defaults to 11
            Minimum size of each columns in the resulting file.

        Returns
        -------
        self : RefractorVelocityField
            RefractorVelocityField unchanged.

        Raises
        ------
        ValueError
            If RefractorVelocityField is empty.
        """
        if self.is_empty:
            raise ValueError("Field is empty. Could not dump empty field.")
        df_list = [calc_df_to_dump(rv) for rv in self.item_container.values()]
        dump_rv(df_list, path=path, encoding=encoding, min_col_size=min_col_size)
        return self

    def plot_fit(self, **kwargs):
        """Plot an interactive map of each parameter of a near-surface velocity model and display an offset-traveltime
        curve with data used to fit the model upon clicking on a map. Can be called only for fields constructed
        directly from first break data.

        Plotting must be performed in a JupyterLab environment with the the `%matplotlib widget` magic executed and
        `ipympl` and `ipywidgets` libraries installed.

        Parameters
        ----------
        figsize : tuple with 2 elements, optional, defaults to (4.5, 4.5)
            Size of the created figures. Measured in inches.
        refractor_velocity_plot_kwargs : dict, optional
            Additional keyword arguments to be passed to `RefractorVelocity.plot`.
        kwargs : misc, optional
            Additional keyword arguments to be passed to `MetricMap.plot`.
        """
        FitPlot(self, **kwargs).plot()
