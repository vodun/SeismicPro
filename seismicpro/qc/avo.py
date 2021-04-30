""" Class for AVO QC """

import os

from seismicpro.batchflow import B
from seismicpro import FieldIndex, CustomIndex
from .avo_dataset import AvoDataset
from .utils import avo_plot, std_plot

class AVOQC():
    """ AVO analysis """

    def __init__(self):
        self._path = None
        self._avo_data = None
        self._fig = None
        self._extra_headers = ['offset', 'CDP', 'CROSSLINE_3D', 'INLINE_3D']
        self._plot_dict = None


    def plot(self, path, bin_size, window, method, plot_type, horizon_path=None):
        """ .!! """
        self._path = path
        name = os.path.splitext(os.path.basename(self._path))[0]

        horizon_window = window if horizon_path is not None else None
        method = 'abs' if method else 'rms'
        plot_type =  'std' if plot_type else 'avo'

        index = CustomIndex(FieldIndex(name='raw', extra_headers=self._extra_headers, path=path),
                                       index_name='CDP')
        if len(index.indices) == 0:
            raise ValueError('Given file either empty or have unreadable headers.')

        dataset = AvoDataset(index)
        batch_size = 10 if len(index.indices) > 10 else len(index.indices)

        hor_pipeline = (dataset.p
                        .load(fmt='segy', components='raw')
                        .sort_traces(src='raw', dst='raw', sort_by='offset')
                        .find_avo_distribution(B(), src='raw', bin_size=bin_size, 
                                               window=window, horizon_path=horizon_path,
                                               horizon_window=horizon_window, method=method,
                                               container_name=name)
                        .run_later(batch_size=batch_size, n_epochs=1, shuffle=False,
                                   drop_last=False, bar=True))
        hor_pipeline.run()
        self._avo_data = getattr(dataset, name)

        self._plot_dict = dict(amp_size=3,
                               avg_size=50,
                               figsize=(15,7),
                               title='AVO for {}'.format(name),
                               is_approx=False,
                               stats='both')

        if plot_type == 'avo':
            self._fig = avo_plot(data=self._avo_data,
                                 bin_size=bin_size,
                                 **self._plot_dict)
        else:
            self._fig = std_plot(avo_results=[self._avo_data],
                                 bin_size=bin_size,
                                 names=[name],
                                 align_mean=False,
                                 **self._plot_dict)
