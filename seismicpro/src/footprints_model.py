""" Footprints attenuation with model """

import os
import tempfile

import h5py

from seismiqb.batchflow.batchflow.models.torch import EncoderDecoder
from seismiqb.batchflow import D, B, V, Pipeline
from seismiqb import SeismicGeometry, SeismicCubeset


def predict_footprints(inp_path, mdl_path, save_res, device='cpu'):
    """ Inference for FP attenuation """

    CROP_LEN = 128
    BATCH_SIZE = 128
    CROP_SHAPE = (CROP_LEN, CROP_LEN, 1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        if inp_path.endswith('.sgy'):
            print("Prepaing input data...")
            geometry = SeismicGeometry(inp_path)
            inp_path = os.path.join(tmpdirname, "tmp_inp.hdf5")
            geometry.make_hdf5(path_hdf5=inp_path, bar=False)

        dataset_test = SeismicCubeset([inp_path])
        dataset_test.load_geometries()
        dataset_test.make_grid(dataset_test.indices[0], CROP_SHAPE, batch_size=BATCH_SIZE*4)

        config_predict = {
            'build': False,
            'load/path': mdl_path,
            'device': device
        }

        inference_init_tmpl = (
            Pipeline()
            .init_variable('result_preds', [])
            .init_model('dynamic', EncoderDecoder, 'model', config=config_predict)
            .crop(points=D('grid_gen')(), shape=CROP_SHAPE)
        )

        load_img_tmpl = (
            Pipeline()
            .load_cubes(src_geometry='geometries', dst='images', slicing='native')
            .scale(mode='q', src='images')
        )

        inference_pipeline = (
            (inference_init_tmpl + load_img_tmpl)
            .transpose(src=['images'], order=(2, 0, 1))
            # Predict with model, then aggregate
            .predict_model('model',
                           B('images'),
                           fetches='predictions',
                           save_to=B('predictions'))
            .update(V('result_preds', mode='e'), B('predictions'))
        ) << dataset_test

        print("Making prediction...")
        inference_pipeline.run(D('size'), n_iters=dataset_test.grid_iters, bar=True)

        assembled_pred = dataset_test.assemble_crops(inference_pipeline.v('result_preds'), order=(1, 2, 0))
        g1 = dataset_test.geometries[dataset_test.indices[0]]

        assembled_pred[g1.zero_traces.nonzero()] = 0
        assembled_pred = assembled_pred * max(abs(g1.q01), abs(g1.q99))

        path_hdf5 = os.path.join(tmpdirname, "tmp.hdf5")

        with h5py.File(path_hdf5, "a") as file_hdf5:
            cube_hdf5 = file_hdf5.create_dataset('cube', g1.cube_shape)
            cube_hdf5[:] = assembled_pred

        print(f"Writing output to {save_res} ...")
        g1.make_sgy(path_hdf5=path_hdf5, path_segy=save_res, zip_result=False)
