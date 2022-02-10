from inspect import signature

import pandas as pd

from ...batchflow import Pipeline


class Metric:
    name = None
    vmin = None
    vmax = None
    is_lower_better = None

    @staticmethod
    def calc(*args, **kwargs):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        _ = args
        for key, val in kwargs.items():
            setattr(self, key, val)


class PlottableMetric(Metric):
    @staticmethod
    def plot(*args, ax, x_ticker, y_ticker, **kwargs):
        raise NotImplementedError

    def coords_to_args(self, coords):
        raise NotImplementedError

    def plot_on_click(self, coords, ax, x_ticker, y_ticker, **kwargs):
        coords_args, coords_kwargs = self.coords_to_args(coords)
        self.plot(*coords_args, ax=ax, x_ticker=x_ticker, y_ticker=y_ticker, **coords_kwargs, **kwargs)


class PipelineMetric(PlottableMetric):
    args_to_unpack = "all"

    def __init__(self, pipeline, calculate_metric_index, coords_to_indices, **kwargs):
        super().__init__(**kwargs)
        self.dataset = pipeline.dataset

        if not isinstance(coords_to_indices, dict) or any(len(indices) > 1 for indices in coords_to_indices.values()):
            # TODO: fallback to filtering the dataset by given coordinates and MapBinPlot
            raise ValueError("Duplicated dataset indices found for some coordinates")
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

    @staticmethod
    def plot(batch, ax, x_ticker, y_ticker, plot_component, **kwargs):
        item = getattr(batch, plot_component)[0]
        item.plot(ax=ax, x_ticker=x_ticker, y_ticker=y_ticker, **kwargs)

    def coords_to_args(self, coords):
        subset = self.dataset.create_subset(self.coords_to_indices[coords])
        batch = (subset >> self.plot_pipeline).next_batch(1, shuffle=False)
        return (batch,), {}


class UnpackingMetric(PipelineMetric):
    def coords_to_args(self, coords):
        (batch,), _ = super().coords_to_args(coords)

        # Get params passed to Metric.calc with possible named expressions and evaluate them
        sign = signature(batch.calculate_metric)
        bound_args = sign.bind(*self.calculate_metric_args, **self.calculate_metric_kwargs)
        bound_args.apply_defaults()
        calc_args = self.plot_pipeline._eval_expr(bound_args.arguments["args"], batch=batch)
        calc_kwargs = self.plot_pipeline._eval_expr(bound_args.arguments["kwargs"], batch=batch)

        args, _ = batch._unpack_metric_args(self, *calc_args, **calc_kwargs)
        return args[0]


def define_metric(cls_name="MetricPlaceholder", base_cls=Metric, **kwargs):
    return type(cls_name, (base_cls,), kwargs)
