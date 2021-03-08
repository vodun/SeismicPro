import sys
from collections import defaultdict

import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

sys.path.append('..')
from seismicpro.src.utils import to_list
from seismicpro.batchflow import Pipeline, V, F, Notifier


sns.set_theme(style="darkgrid")
def benchmark_method(method, dataset, targets, batch_sizes, n_iters=10, root_ppl=None, notifier_monitors=None,
                     figsize=(15, 7), **method_kwargs):
    batch_sizes = to_list(batch_sizes)
    targets = to_list(targets)

    root_pipeline = Pipeline()
    if root_ppl is not None:
        if isinstance(root_ppl, str):
            root_pipeline += Pipeline().load(src='raw', fmt='segy')
            if root_ppl == 'sort':
                root_pipeline += Pipeline().sort(src='raw', dst='raw', by='offset')
            elif root_ppl != 'load':
                raise ValueError('Wrong root_ppl name.')
        elif not isinstance(root_ppl, Pipeline):
            raise ValueError('Wrong root_ppl type.')
        else:
            root_pipeline += root_ppl

    times_dict = dict()
    notifiers_dict = dict()
    for target in tqdm(targets):
        for batch_size in tqdm(batch_sizes):
            step_times, notifiers = _benchmark_step(method=method, dataset=dataset, root_pipeline=root_pipeline,
                                                    target=target, batch_size=batch_size, n_iters=n_iters,
                                                    notifier_monitors=notifier_monitors, **method_kwargs)
            step_name = f'{target}_{batch_size}'
            times_dict[step_name] = step_times

            for notifier in notifiers:
                if notifier['name'] not in notifiers_dict.keys():
                    notifiers_dict[notifier['name']] = dict(ylabel=notifier['source'].UNIT)
                notifiers_dict[notifier['name']][f'{step_name}'] = notifier['data']

    benchmark_plotter(res_dict=times_dict, title=method.__name__, ylabel='Time (s)', figsize=figsize)
    for monitor_name, results in notifiers_dict.items():
        ylabel = results.pop('ylabel')
        benchmark_plotter(res_dict=results, title=monitor_name, ylabel=ylabel, figsize=figsize)
    return times_dict, notifiers_dict

def _benchmark_step(method, dataset, root_pipeline, target, batch_size, n_iters, notifier_monitors, **kwargs):
    kwargs.update(target=target)
    method_pipeline = getattr(Pipeline(), method.__name__)(**kwargs)
    main_pipeline = (root_pipeline + method_pipeline) << dataset
    main_pipeline.run(batch_size, n_iters=n_iters, shuffle=42, drop_last=False, bar=False, profile=True,
                      notifier=Notifier(monitors=notifier_monitors))

    action_name = method.__name__ + f' #{main_pipeline.num_actions-1}'
    time_df = main_pipeline.show_profile_info(per_iter=True, detailed=False)
    time_vals = time_df['total_time'].loc[:, action_name].values

    notifier = main_pipeline.notifier.data_containers
    return time_vals, notifier


def benchmark_plotter(res_dict, title=None, ylabel=None, figsize=None):
    plt.figure(figsize=figsize)
    target_dict = defaultdict(list)
    iters = []

    for bench_name, res in res_dict.items():
        target_name, batch_size = bench_name.split('_')
        target_dict[target_name].append(res)
        iters.append(int(batch_size))
    iters = np.unique(iters)

    for target, times in target_dict.items():
        x_axis = np.concatenate([[ix] * len(time) for ix, time in enumerate(times)])
        sns.lineplot(x=x_axis, y=np.concatenate(times), label=target, ci='sd')

    plt.xticks(ticks=np.arange(len(iters)), labels=iters)
    plt.xlabel('Batch size')
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
