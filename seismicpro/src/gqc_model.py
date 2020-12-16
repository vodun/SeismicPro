""" Inference script that takes segy file and Geometry model.
Predicts whether the gather has geometry erorr - flag 1, or no error - flag 0.
Finally dumps results to csv file.
Note: works only on the gathers containing atleast 100 traces.
"""
import os
import sys
import argparse

import numpy as np


from seismicpro.batchflow import B, Pipeline
from seismicpro.batchflow.models.torch import SEResNet34 # pylint: disable=import-error
from seismicpro.src import CustomIndex, SeismicDataset
from .gqc_utils import GeomSeismicDataset, GeomCustomIndex

def make_prediction():
    """ Read the model and data paths and run inference pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_raw', type=str, help="Path to SEGY file.", required=True)
    parser.add_argument('-m', '--path_model', type=str, help="Path to trained model.", required=True)
    parser.add_argument('-d', '--path_dump', type=str, help="Path to CSV file where \
                        the results would be stored.", default='geometry.csv')
    parser.add_argument('-bs', '--batch_size', type=int, help="The number of gathers in the batch.", default=100)
    parser.add_argument('-v', '--velocity', type=int, help="LMO velocity, km / sec.", default=1.6)
    parser.add_argument('-ts', '--trace_len', type=int, help="The number of first samples  of the trace to load.", default=100)
    parser.add_argument('-dvc', '--device', type=str, help="The device for \
                        inference. Can be 'cpu' or 'gpu'.", default='cpu')
    args = parser.parse_args()
    path_raw = args.path_raw
    model = args.path_model
    save_to = args.path_dump
    batch_size = args.batch_size
    velocity = args.velocity
    trace_len = args.trace_len
    device = args.device
    predict(path_raw, model, save_to, batch_size, velocity, trace_len, device)

def predict(path_raw, path_model, save_to, batch_size, velocity, trace_len, device):
    """Make predictions and dump results using loaded model and path to data.
    Parameters
    ----------
    path_raw: str
        Path to SEGY file.
    path_model: str
        Path to the file with trained model.
    save_to: str, default: 'dump.csv'
        Path to CSV file where the results will be stored.
    bs: int, default: 1000
        The batch size for inference.
    trace_len: int, default 100
        The number of time samples in the crop.
    device: str or torch.device, default: 'cpu'
        The device used for inference. Can be 'gpu' in case of avaliavle GPU.
    """
    index = GeomCustomIndex(name='raw', path=path_raw, index_name='RecieverID',
                            extra_headers=['offset', 'GroupX', 'GroupY'])

    mask = index.tracecounts > 100
    index = index.create_subset(index.indices[mask])

    index.sort(by='offset')
    index.keep_first(slice(0, 100))

    data = GeomSeismicDataset(index)

    config_predict = {
        'build': False,
        'load/path': path_model,
        'device': device
    }

    try:
        os.remove(save_to)
    except OSError:
        pass

    inference_pipeline = (
        data.p
        .init_model('dynamic', SEResNet34, 'geom_class', config=config_predict)
        .load(components='raw', fmt='segy')
        .apply_parallel(lambda x: np.clip(x, -1, 1), src='raw', dst='raw')
        .load(fmt='index', components=('offset', 'GroupX', 'GroupY'),
              src=('offset', 'GroupX', 'GroupY'))
        .apply_parallel(lambda x: np.abs(x), src='offset', dst='offset')
        .LMO(V=velocity, length=trace_len, src_traces='raw', dst='lmo', pad=20)
        .call(lambda x: np.stack(x.lmo)[:, np.newaxis, :, :], save_to=B('lmo'))
        .predict_model('geom_class', B('lmo'), fetches='predictions', save_to=B('pred'))
        .call(lambda x: np.argmax(x.pred, axis=-1), save_to=B('pred_class'))
        .dump(src='pred_class', path='geometry.csv', columns=('GroupX', 'GroupY'), fmt='geom')
        .run_later(batch_size, n_epochs=1, drop_last=False, shuffle=True, bar=True)
    )

    inference_pipeline.run()

if __name__ == "__main__":
    sys.exit(make_prediction())
