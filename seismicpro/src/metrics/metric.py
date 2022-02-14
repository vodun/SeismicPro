import warnings
from inspect import signature
from functools import partial

import pandas as pd

from .interactive_plot import ScatterMapPlot, BinarizedMapPlot
from ...batchflow import Pipeline


class Metric:
    name = None
    min_value = None
    max_value = None
    is_lower_better = None

    @staticmethod
    def calc(*args, **kwargs):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        _ = args
        for key, val in kwargs.items():
            setattr(self, key, val)


class PlottableMetric(Metric):
    vmin = None
    vmax = None
    interactive_scatter_map_class = ScatterMapPlot
    interactive_binarized_map_class = BinarizedMapPlot

    def plot_on_click(self, coords, ax, x_ticker, y_ticker, **kwargs):
        raise NotImplementedError


class PipelineMetric(PlottableMetric):
    args_to_unpack = "all"

    def __init__(self, pipeline, calculate_metric_index, coords_cols, coords_to_indices, **kwargs):
        super().__init__(**kwargs)
        self.dataset = pipeline.dataset
        self.coords_dataset = self.dataset.copy().reindex(coords_cols)

        self.coords_cols = coords_cols
        self.coords_to_indices = coords_to_indices

        # Slice the pipeline in which the metric was calculated up to its calculate_metric call
        calculate_metric_indices = [i for i, action in enumerate(pipeline._actions)
                                      if action["name"] == "calculate_metric"]
        calculate_metric_action_index = calculate_metric_indices[calculate_metric_index]
        actions = pipeline._actions[:calculate_metric_action_index]
        self.plot_pipeline = Pipeline(pipeline=pipeline, actions=actions)

        # Get args and kwargs of the calculate_metric call with possible named expressions in them
        self.calculate_metric_args = pipeline._actions[calculate_metric_action_index]["args"]
        self.calculate_metric_kwargs = pipeline._actions[calculate_metric_action_index]["kwargs"]

    @staticmethod
    def calc(metric):
        return metric

    def gen_batch(self, coords, batch_src):
        if batch_src not in {"index", "coords"}:
            raise ValueError("Unknown source to get the batch from. Available options are 'index' and 'coords'.")
        if batch_src == "index":
            if self.coords_to_indices is None:
                raise ValueError("Unable to use indices to get the batch by coordinates since they were not passed "
                                 "during metric instantiation. Please specify batch_src='coords'.")
            subset = self.dataset.create_subset(self.coords_to_indices[coords])
        else:
            indices = []
            for concat_id in range(self.coords_dataset.index.next_concat_id):
                index_candidate = (concat_id,) + coords
                if index_candidate in self.coords_dataset.index.index_to_headers_pos:
                    indices.append(index_candidate)
            subset = self.coords_dataset.create_subset(pd.MultiIndex.from_tuples(indices))

        if len(subset) > 1:
            # TODO: try moving to MapBinPlot in this case
            warnings.warn("Multiple gathers exist for given coordinates, only the first one is shown", RuntimeWarning)
        return subset.next_batch(1, shuffle=False)

    @staticmethod
    def plot_component(batch, plot_component, ax, x_ticker, y_ticker, **kwargs):
        item = getattr(batch, plot_component)[0]
        item.plot(ax=ax, x_ticker=x_ticker, y_ticker=y_ticker, **kwargs)

    def get_calc_args(self, batch):
        # Get params passed to Metric.calc with possible named expressions and evaluate them
        sign = signature(batch.calculate_metric)
        bound_args = sign.bind(*self.calculate_metric_args, **self.calculate_metric_kwargs)
        bound_args.apply_defaults()
        calc_args = self.plot_pipeline._eval_expr(bound_args.arguments["args"], batch=batch)
        calc_kwargs = self.plot_pipeline._eval_expr(bound_args.arguments["kwargs"], batch=batch)
        args, _ = batch._unpack_metric_args(self, *calc_args, **calc_kwargs)
        return args[0]

    @staticmethod
    def plot(*args, ax, x_ticker, y_ticker, **kwargs):
        raise NotImplementedError("Specify plot_component argument since plot method is not overridden")

    def plot_on_click(self, coords, ax, x_ticker, y_ticker, batch_src="index", pipeline=None, plot_component=None,
                      **kwargs):
        # Generate a batch by given coordinated and source
        batch = self.gen_batch(coords, batch_src)

        # Execute passed pipeline for the batch
        default_pipelines = {
            "index": self.plot_pipeline,
            "coords": Pipeline().load(src=plot_component)
        }
        if pipeline is None:
            pipeline = default_pipelines[batch_src]
        batch = pipeline.execute_for(batch)

        # Plot either passed component or args used for metric calculation
        if plot_component is not None:
            self.plot_component(batch, plot_component, ax, x_ticker, y_ticker, **kwargs)
        else:
            coords_args, coords_kwargs = self.get_calc_args(batch)
            self.plot(*coords_args, ax=ax, x_ticker=x_ticker, y_ticker=y_ticker, **coords_kwargs, **kwargs)


class PartialMetric:
    def __init__(self, metric, *args, **kwargs):
        if not(isinstance(metric, PartialMetric) or isinstance(metric, type) and issubclass(metric, Metric)):
            raise ValueError("metric must be either an instance of PartialMetric or a subclass of Metric")
        self.metric = partial(metric, *args, **kwargs)

    def __getattr__(self, name):
        if name in self.metric.keywords:
            return self.metric.keywords[name]
        return getattr(self.metric.func, name)

    def __call__(self, *args, **kwargs):
        return self.metric(*args, **kwargs)


def is_metric(obj):
    return isinstance(obj, (Metric, PartialMetric)) or isinstance(obj, type) and issubclass(obj, Metric)


def define_metric(cls_name="MetricPlaceholder", base_cls=Metric, **kwargs):
    return type(cls_name, (base_cls,), kwargs)
