"""Implements Benchmark class"""
# pylint: disable=import-error

import dill
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from seismicpro.src.utils import to_list
from seismicpro.batchflow import Pipeline, CPUMonitor, C
from seismicpro.batchflow.research import Option, Research, EC


sns.set_theme(style="darkgrid")
class Benchmark:
    """A class aimed to find an optimal parallelization engine for methods decorated with
    :func:`~decorators.batch_method`.

    `Benchmark` runs experiments with all combinations of given parallelization engines (`targets`) and batch sizes for
    the specified method and measure the time for all repetitions. To get a more accurate time estimation, each
    experiment is run `n_iters` times. In the result, the graph with relations between elapsed time and batch size are
    plotted for every `target`.

    Simple usage of `Benchmark` contains three steps:
    1. Define Benchmark instance.
    2. Call `benchmark.run()` to run the benchmark.
    3. Call `benchmark.plot()` to plot the results.

    Parameters
    ----------
    method_name : str
        A name of the benchmarked method.
    method_kwargs : dict
        Additional keyword arguments to the benchmarked method.
    targets : str or array of str
        Name(s) of target from :func:`~batchflow.batchflow.decorators.inbatch_parallel`.
    batch_sizes : int or array of int
        Batch size(s) on which the benchmark is performed.
    dataset : Dataset
        Dataset on which the benchmark is conducted.
    root_pipeline : Pipeline, optional, by default None
        Pipeline that contains actions to be performed before the benchmarked method.

    Attributes
    ----------
    method_name : str
        A name of the benchmarked method.
    targets : str or array of str
        Name(s) of targets from :func:`~batchflow.batchflow.decorators.inbatch_parallel`.
    batch_sizes : int or array of int
        Batch size(s) on which the benchmark is performed.
    results : None or pd.DataFrame
        A DataFrame with benchmark results.
    template_pipeline : Pipeline
        Pipeline that contains `root_pipeline`, benchmarked method, and dataset.
    """
    def __init__(self, method_name, method_kwargs, targets, batch_sizes, dataset, root_pipeline=None):
        self.method_name = method_name
        self.targets = to_list(targets)
        self.batch_sizes = to_list(batch_sizes)
        self.results = None

        # Add benchmarked method to the `root_pipeline` with `method_kwargs` and `target` from config.
        method_kwargs['target'] = C('target')
        root_pipeline = Pipeline() if root_pipeline is None else root_pipeline
        root_pipeline = getattr(root_pipeline, self.method_name)(**method_kwargs)
        self.template_pipeline = root_pipeline << dataset

        # Run the pipeline once to precompile all numba callables
        self._warmup()

    def _warmup(self):
        """Run `self.template_pipeline` once."""
        (self.template_pipeline << {'target': 'for'}).next_batch(1)

    def save(self, path):
        """Pickle Benchmark to a file.

        Parameters
        ----------
        path : str
            A path to save the resulted benchmark.

        Returns
        -------
        self : Benchmark
            Unchanged Benchmark.
        """
        with open(path, 'wb') as file:
            dill.dump(self, file)
        return self

    @staticmethod
    def load(path):
        """Unpickle Benchmark from a file.

        Parameters
        ----------
        path : str
            Path to pickled file.

        Returns
        -------
        benchmark : Benchmark
            Unpickled Benchmark.
        """
        with open(path, 'rb') as file:
            benchmark = dill.load(file)
        return benchmark

    def run(self, n_iters=10, shuffle=False, bar=False, env_meta=False):
        """Run benchmark.

        The method measures the execution time and CPU utilization of the benchmarked method with all combinations of
        targets and batch sizes.

        Parameters
        ----------
        n_iters : int, optional, by default 10
            The number of iterations for the method with specified parameters.
        shuffle : int or bool, by default False
            Specifies the randomization in the pipeline.
            If `False`: items go sequentially, one after another as they appear in the index;
            If `True`: items are shuffled randomly before each epoch;
            If int: a seed number for a random shuffle;
        bar : bool, optional, default False
            Whether to use progress bar or not.
        env_meta : dict or bool, optional, default False
            if dict, kwargs for :meth:`~batchflow.batchflow.research.attach_env_meta`
            if bool, whether to attach environment meta or not.

        Returns
        -------
        self : Benchmark
            Benchmark with computed reuslts.
        """
        domain = Option('target', self.targets) * Option('batch_size', self.batch_sizes)

        # Create research that will run pipeline with different parameters based on given `domain`
        research = (Research(domain=domain, n_reps=1)
            .add_callable(self._run_single_pipeline, config=EC(), n_iters=n_iters, shuffle=shuffle,
                          save_to=['Time', 'CPUMonitor'])
        ).run(n_iters=1, dump_results=False, parallel=False, workers=1, bar=bar, env_meta=env_meta)

        # Load benchmark's results.
        self.results = research.results.to_df().set_index(['target', 'batch_size'])[['Time', 'CPUMonitor']]
        return self

    def _run_single_pipeline(self, config, n_iters, shuffle):
        """Benchmark method with a particular `batch_size` and `target`."""
        pipeline = self.template_pipeline << config
        with CPUMonitor() as cpu_monitor:
            pipeline.run(C('batch_size'), n_iters=n_iters, shuffle=shuffle, drop_last=True, notifier=False,
                         profile=True)

        # Processing the results for time costs.
        time_df = pipeline.show_profile_info(per_iter=True, detailed=False)
        action_name = f'{self.method_name} #{pipeline.num_actions-1}'
        run_time = time_df['total_time'].loc[:, action_name].values

        return run_time, cpu_monitor.data

    def plot(self, figsize=(15, 7), cpu_util=False):
        """Plot a graph of time verus `batch_size` for every `target`.

        Each point on the graph shows the average value and the standard deviation of elapsed time over `n_iters`
        iterations.

        Parameters
        ----------
        figsize : tuple, optional, by default (15, 7)
            Output plot size.
        cpu_util : bool
            If True, the CPU utilization is plotted after the elapsed time plot.
        """
        results = self.results.drop(columns='CPUMonitor') if not cpu_util else self.results
        for col_name, col_series in results.iteritems():
            sub_df = col_series.explode().reset_index()

            plt.figure(figsize=figsize)
            sns.lineplot(data=sub_df, x='batch_size', y=col_name, hue='target', ci='sd', marker='o')

            plt.title(f"{col_name} for {self.method_name} method")
            plt.xticks(ticks=self.batch_sizes, labels=self.batch_sizes)

            plt.xlabel('Batch size')
            plt.ylabel('Time (s)' if col_name == 'Time' else 'CPU (%)')
        plt.show()
