from inspect import signature

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

    def gen_batch(self, coords, batch_src):
        if batch_src == "index":
            return self.dataset.create_subset(self.coords_to_indices[coords]).next_batch(1, shuffle=False)
        if batch_src == "coords":
            # TODO: get the gather corresponding to the coords from the reindexed dataset
            raise ValueError
        raise ValueError("Unknown source to get the batch from")

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
    def plot_calc_args(*args, ax, x_ticker, y_ticker, **kwargs):
        raise NotImplementedError("Specify plot_component argument since plot_calc_args method is not overridden")

    def plot_on_click(self, coords, ax, x_ticker, y_ticker, batch_src="index", pipeline=None, plot_component=None,
                      **kwargs):
        batch = self.gen_batch(coords, batch_src)
        if pipeline is None:
            pipeline = self.plot_pipeline
        batch = pipeline.execute_for(batch)
        if plot_component is not None:
            self.plot_component(batch, plot_component, ax, x_ticker, y_ticker, **kwargs)
        else:
            coords_args, coords_kwargs = self.get_calc_args(batch)
            self.plot_calc_args(*coords_args, ax=ax, x_ticker=x_ticker, y_ticker=y_ticker, **coords_kwargs, **kwargs)


def define_metric(cls_name="MetricPlaceholder", base_cls=Metric, **kwargs):
    return type(cls_name, (base_cls,), kwargs)
