import os
import sys
import shutil
from hashlib import sha1

import dill
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from seismicpro.src.utils import to_list
from seismicpro.batchflow import Pipeline, CPUMonitor, C
from seismicpro.batchflow.research import Option, Research, Results, RC


sns.set_theme(style="darkgrid")
class Benchmark:
    def __init__(self, method_name, method_kwargs, targets, batch_sizes, dataset, n_iters=10,
                 root_pipeline=None, benchmark_cpu=True, save_to=None):
        self.method_name = method_name
        self.targets = to_list(targets)
        self.batch_sizes = to_list(batch_sizes)
        self.n_iters = n_iters
        self.benchmark_names = ['Time', 'CPUMonitor']
        self.results = None
        research_name = '_'.join([self.method_name, *self.targets, *list(map(str, self.batch_sizes))])
        name_hash = sha1()
        name_hash.update(research_name.encode('utf-8'))
        self.research_name = 'research_' + str(name_hash.hexdigest())

        self.benchmark_cpu = benchmark_cpu

        self.root_pipeline = Pipeline()
        if root_pipeline is not None:
            self.root_pipeline += root_pipeline

        method_pipeline = getattr(Pipeline(), self.method_name)(target=C('target'), **method_kwargs)
        self.template_pipeline = (self.root_pipeline + method_pipeline) << dataset
        self.save_to = save_to

        # If research goes wrong, delete directory before run the new one
        self._clear_previous_results()

    def save_benchmark(self):
        with open(self.save_to, 'wb') as file:
            dill.dump(self, file)
        return self

    @staticmethod
    def load_benchmark(path):
        with open(path, 'rb') as file:
            benchmark = dill.load(file)
        return benchmark

    def _clear_previous_results(self):
        if os.path.exists(self.research_name):
            shutil.rmtree(self.research_name)

    def _warmup(self):
        (self.template_pipeline << {'target': 'for'}).next_batch(1)

    def run(self):
        domain = Option('target', self.targets) * Option('batch_size', self.batch_sizes)

        # Run the pipeline once to precompile all numba callables
        self._warmup()

        # Create research that will run pipeline with different parameters based on given `domain`
        research = (Research()
            .init_domain(domain, n_reps=1)
            .add_callable(self.run_single_pipeline, config=RC().config(), returns=self.benchmark_names)
        )
        research.run(n_iters=1, name=self.research_name, bar=True, workers=1)

        # Load benchmark's results.
        results_df = Results(self.research_name).df
        # Processing dataframe to a more convenient view.
        results_df = results_df.astype({'batch_size': 'int32'})
        results_df.sort_values(by='batch_size', inplace=True)
        results_df.set_index(['target', 'batch_size'], inplace=True)
        self.results = results_df[self.benchmark_names]

        if self.save_to is not None:
            self.save_benchmark()
        self._clear_previous_results()
        return self

    def run_single_pipeline(self, config):
        pipeline = self.template_pipeline << config
        with CPUMonitor() as cpu_monitor:
            pipeline.run(C('batch_size'), n_iters=self.n_iters, shuffle=42, drop_last=True, bar=False, profile=True)

        # Processing the results for time costs.
        time_df = pipeline.show_profile_info(per_iter=True, detailed=False)
        action_name = f'{self.method_name} #{pipeline.num_actions-1}'
        run_time = time_df['total_time'].loc[:, action_name].values

        # Processing the results for CPUmonitor.
        cpu_util = cpu_monitor.data if self.benchmark_cpu is not None else None
        return run_time, cpu_util

    def plot(self, figsize=(15, 7)):
        for col_name, column in self.results.iteritems():
            plt.figure(figsize=figsize)
            sub_df = column.reset_index()
            for target, df in sub_df.groupby('target'):
                items = df[col_name].values
                items_batchsize = np.concatenate([[bs] * len(val) for bs, val in zip(df['batch_size'], items)])
                sns.lineplot(x=items_batchsize, y=np.concatenate(items), label=target, ci='sd', marker='o')

            plt.title(f"{col_name} for {self.method_name} method")
            plt.xticks(ticks=self.batch_sizes, labels=self.batch_sizes)

            plt.xlabel('Batch size')
            plt.ylabel('Time (s)' if col_name == 'Time' else '%')
        plt.show()
