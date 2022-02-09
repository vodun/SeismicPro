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
        _ = args, kwargs

    def finalize(self, *args, **kwargs):
        _ = args, kwargs


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

    def __init__(self, pipeline, calculate_metric_index):
        self.pipeline = pipeline
        self.calculate_metric_index = calculate_metric_index

    def finalize(self, coords_to_indices):
        self.coords_to_indices = coords_to_indices
        calculate_metric_indices = [i for i, action in enumerate(self.pipeline._actions)
                                      if action["name"] == "calculate_metric"]
        calculate_metric_action_index = calculate_metric_indices[self.calculate_metric_index]
        actions = self.pipeline._actions[:calculate_metric_action_index]
        self.plot_pipeline = Pipeline(pipeline=self.pipeline, actions=actions)
        self.calculate_metric_params = self.pipeline._actions[calculate_metric_action_index]

    def coords_to_args(self, coords):
        subset_index = pd.MultiIndex.from_tuples([self.coords_to_indices[coords],])
        batch = (self.pipeline.dataset.create_subset(subset_index) >> self.plot_pipeline).next_batch(1, shuffle=False)
        return (batch,), {}


class PrecalculatedMetric(PipelineMetric):
    @staticmethod
    def calc(metric):
        return metric

    @staticmethod
    def plot(batch, ax, x_ticker, y_ticker, plot_component, **kwargs):
        item = getattr(batch, plot_component)[0]
        item.plot(ax=ax, x_ticker=x_ticker, y_ticker=y_ticker, **kwargs)


class UnpackingMetric(PipelineMetric):
    def coords_to_args(self, coords):
        (batch,), _ = super().coords_to_args(coords)

        # Get _unpack_metric_args params with possible named expressions and evaluate them
        sign = signature(batch.calculate_metric)
        bound_args = sign.bind(*self.calculate_metric_params["args"], **self.calculate_metric_params["kwargs"])
        bound_args.apply_defaults()
        calc_args = self.pipeline._eval_expr(bound_args.arguments["args"], batch=batch)
        calc_kwargs = self.pipeline._eval_expr(bound_args.arguments["kwargs"], batch=batch)

        args, _ = batch._unpack_metric_args(self, *calc_args, **calc_kwargs)
        return args[0]


def define_metric(cls_name="MetricPlaceholder", base_cls=Metric, **kwargs):
    return type(cls_name, (base_cls,), kwargs)
