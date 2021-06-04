import os
import sys
import shutil
from hashlib import sha1

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from seismicpro.src.utils import to_list
from seismicpro.batchflow import Pipeline, CPUMonitor, C
from seismicpro.batchflow.research import Option, Research, Results, RC


sns.set_theme(style="darkgrid")
class Benchmark:
    def __init__(self, method_name, method_kwargs, targets, batch_sizes, dataset, dataset_name='raw', n_iters=10,
                 root_pipeline=None, benchmark_cpu=False, delete_research=False):
        self.method_name = method_name
        self.targets = to_list(targets)
        self.batch_sizes = to_list(batch_sizes)
        self.dataset = dataset
        self.dataset_name = dataset_name
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
        self.template_pipeline = (self.root_pipeline + method_pipeline) << self.dataset
        self.delete_research = delete_research

    def run(self, **method_kwargs):
        domain = Option('target', self.targets) * Option('batch_size', self.batch_sizes)
        self._clear_previous_results()

        # Create research that will run pipeline with different parameters based on given `domain`.
        research = (Research()
            .init_domain(domain, n_reps=1)
            .add_callable(self.run_single_pipeline, config=RC().config(), returns=self.benchmark_names,
                          **method_kwargs)
        )
        research.run(n_iters=1, name=self.research_name, bar=True, workers=1)

        # Load benchmark's results.
        results_df = Results(self.research_name).df
        # Processing dataframe to a more convenient view.
        results_df = results_df.astype({'batch_size': 'int32'})
        results_df.sort_values(by='batch_size', inplace=True)
        results_df.set_index(['target', 'batch_size'], inplace=True)
        self.results = results_df[self.benchmark_names]
        if self.delete_research:
            self._clear_previous_results()
        return self

    def run_single_pipeline(self, config, **method_kwargs):
        pipeline = self.template_pipeline << config
        with CPUMonitor() as cpu_monitor:
            pipeline.run(C('batch_size'), n_iters=self.n_iters, shuffle=42, drop_last=True, bar=False, profile=True)

        # Processing the results for time costs.
        action_name = self.method_name + f' #{pipeline.num_actions-1}'
        time_df = pipeline.show_profile_info(per_iter=True, detailed=False)
        time_vals = time_df['total_time'].loc[:, action_name].values

        # Processing the results for notifier monitors.
        cpu_notifier = cpu_monitor.data if self.benchmark_cpu is not None else None
        return time_vals, cpu_notifier

    def plot(self, figsize=(15, 7)):
        for col_name, column in self.results.iteritems():
            if column.isna().sum() > 0:
                continue
            plt.figure(figsize=figsize)
            sub_df = column.reset_index()
            for target, df in sub_df.groupby('target'):
                items = df[col_name].values
                x_axis = np.concatenate([[ix] * len(time) for ix, time in enumerate(items)])
                sns.lineplot(x=x_axis, y=np.concatenate(items), label=target, ci='sd')

            plt.title(f"{col_name} for {self.method_name} method")
            plt.xticks(ticks=np.arange(len(self.batch_sizes)), labels=self.batch_sizes)

            plt.xlabel('Batch size')
            ylabel = 'Time (s)' if col_name == 'Time' else '%'
            plt.ylabel(ylabel)
            plt.show()
        return self

    def _clear_previous_results(self):
        if os.path.exists(self.research_name):
            shutil.rmtree(self.research_name)
