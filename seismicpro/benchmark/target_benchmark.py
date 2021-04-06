import sys
from collections import defaultdict

import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from seismicpro.src.utils import to_list
from seismicpro.batchflow import Pipeline, V, F, Notifier


sns.set_theme(style="darkgrid")
def benchmark_method(method, dataset, targets, batch_sizes, n_iters=10, root_ppl=None, notifier_monitors=None,
                     figsize=(15, 7), **method_kwargs):
    batch_sizes = to_list(batch_sizes)
    targets = to_list(targets)

    root_pipeline = Pipeline()
    if root_ppl is not None:
        if root_ppl in ['load', 'sort']:
            root_pipeline += Pipeline().load(src='raw', fmt='segy')
            if root_ppl == 'sort':
                root_pipeline += Pipeline().sort(src='raw', dst='raw', by='offset')
        elif isinstance(root_ppl, Pipeline):
            root_pipeline += root_ppl
        else:
            raise ValueError('Wrong root_ppl type.')


    method_results = defaultdict(list)
    notifiers_results = defaultdict(list)
    for target in tqdm(targets):
        for batch_size in tqdm(batch_sizes):
            step_times, notifiers = _benchmark_step(method=method, dataset=dataset, root_pipeline=root_pipeline,
                                                    target=target, batch_size=batch_size, n_iters=n_iters,
                                                    notifier_monitors=notifier_monitors, **method_kwargs)
            method_results[target].append(step_times)

            for notifier in notifiers:
                if notifier['name'] not in notifiers_results.keys():
                    notifiers_results[notifier['name']] = defaultdict(list, ylabel=notifier['source'].UNIT)
                notifiers_results[notifier['name']][target].append(notifier['data'])

    benchmark_plotter(res_dict=method_results, title=method.__name__, ylabel='Time (s)',
                      xticks=batch_sizes, figsize=figsize)
    for monitor_name, results in notifiers_results.items():
        ylabel = results.pop('ylabel')
        benchmark_plotter(res_dict=results, title=monitor_name, ylabel=ylabel, xticks=batch_sizes, figsize=figsize)
    return method_results, notifiers_results

def _benchmark_step(method, dataset, root_pipeline, target, batch_size, n_iters, notifier_monitors, **kwargs):
    kwargs.update(target=target)
    method_pipeline = getattr(Pipeline(), method.__name__)(**kwargs)
    main_pipeline = (root_pipeline + method_pipeline) << dataset
    main_pipeline.run(batch_size, n_iters=n_iters, shuffle=42, drop_last=False, bar=False, profile=True,
                      notifier=Notifier(monitors=notifier_monitors))

    # We assume that our method is the last action in pipeline.
    action_name = method.__name__ + f' #{main_pipeline.num_actions-1}'
    time_df = main_pipeline.show_profile_info(per_iter=True, detailed=False)
    time_vals = time_df['total_time'].loc[:, action_name].values

    notifier = main_pipeline.notifier.data_containers
    return time_vals, notifier


def benchmark_plotter(res_dict, title=None, ylabel=None, xticks=None, figsize=None):
    plt.figure(figsize=figsize)

    for target, times in res_dict.items():
        x_axis = np.concatenate([[ix] * len(time) for ix, time in enumerate(times)])
        sns.lineplot(x=x_axis, y=np.concatenate(times), label=target, ci='sd')

    plt.xlabel('Batch size')
    if xticks:
        plt.xticks(ticks=np.arange(len(xticks)), labels=xticks)
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
