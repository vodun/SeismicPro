""" Test for cropping from arrays with seismic data and assembling crops"""

import os

import pytest
import numpy as np


from seismicpro.batchflow import V, P, B
from seismicpro.src import SeismicDataset, FieldIndex, TraceIndex


PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../datasets/demo_data/teapot_dome_10.sgy')
# 10 field records with shapes (912, 2049) or (966, 2049)

PARAMS = {
    # (index_type, batch_size): crop_shapes
    (FieldIndex, 5): [(100, 150), (12, 100), (12, 2000), (912, 2000), (1000, 2100)],
    (TraceIndex, 100): [(1, 150), (1, 100), (1, 2000), (1, 2100)]
}

PARAMS_LIST = [(*it_bs, cs) for it_bs in PARAMS for cs in PARAMS[it_bs]]


class TestCropAssemble:
    """ Check croppping and assembling """

    @pytest.mark.parametrize('single_iteration', [True, pytest.param(False, marks=pytest.mark.slow)])
    @pytest.mark.parametrize('index_type,batch_size,crop_shape', PARAMS_LIST)
    @pytest.mark.parametrize('assemble_fill_value', [0, 0.5])
    def test_crop_assemble(self, index_type, batch_size, crop_shape, single_iteration, assemble_fill_value):
        """
        Make crops that cover whole array using regular grid,
        then assemble those crops and
        check that the result equals the original array.

        Checks int coords, using different coords for each item in a batch (P named exression), and assembling crops
        """
        index = index_type(path=PATH, name='raw')

        ppl = (
            SeismicDataset(index).p
            .init_variable('raw', default=[])
            .init_variable('assemble', default=[])
            .load(components='raw', fmt='segy', tslice=slice(2000))
            .update(V('raw', 'a'), B('raw'))
            .make_grid_for_crops(src='raw', dst='coords', shape=crop_shape, drop_last=False)
            .crop(src='raw', dst='crops', coords=P(B('coords')), shape=crop_shape, pad_zeros=True)
            .assemble_crops(src='crops', dst='assemble', fill_value=assemble_fill_value)
            .update(V('assemble', 'a'), B('assemble'))
            )

        if single_iteration:
            ppl.run(batch_size, n_iters=1, shuffle=True)
        else:
            ppl.run(batch_size, n_epochs=1, shuffle=False)

        raw_batches_list = ppl.get_variable('raw')
        assemble_batches_list = ppl.get_variable('assemble')

        for raw_list, assemble_list in zip(raw_batches_list, assemble_batches_list):
            for raw, assemble in zip(raw_list, assemble_list):
                assert np.allclose(raw, assemble)


    def test_wrong_action_order(self):
        """ assembling and plotting crops should fail if no cropping was done"""

        index = FieldIndex(path=PATH, name='raw')

        ppl = (
            SeismicDataset(index).p
            .load(components='raw', fmt='segy')
            .assemble_crops(src='raw', dst='assemble')
            )

        with pytest.raises(Exception):
            ppl.run(5, n_iters=1)

        ppl = SeismicDataset(index).p.load(components='raw', fmt='segy')

        batch = ppl.next_batch(5)

        with pytest.raises(Exception):
            batch.crops_plot('raw', index.indices[0])


    @pytest.mark.parametrize('single_iteration', [True, pytest.param(False, marks=pytest.mark.slow)])
    @pytest.mark.parametrize('index_type,batch_size', list(PARAMS.keys()))
    def test_crop_float_coords_ok(self, index_type, batch_size, single_iteration):
        """ Make crops using float coords """

        index = index_type(path=PATH, name='raw')

        ppl = (
            SeismicDataset(index).p
            .init_variable('raw', default=[])
            .init_variable('crops', default=[])
            .load(components='raw', fmt='segy', tslice=slice(2000))
            .update(V('raw', 'a'), B('raw'))
            .crop(src='raw', dst='crops', coords=[(0, 0), (0, 0.5)], shape=(1, 1), pad_zeros=False)
            .update(V('crops', 'a'), B('crops'))
            )

        if single_iteration:
            ppl.run(batch_size, n_iters=1, shuffle=True)
        else:
            ppl.run(batch_size, n_epochs=1, shuffle=False)

        raw_batches_list = ppl.get_variable('raw')
        crops_batches_list = ppl.get_variable('crops')

        for raw_list, crops_list in zip(raw_batches_list, crops_batches_list):
            for raw, crops in zip(raw_list, crops_list):
                assert np.allclose(raw[0, 0], crops[0])
                assert np.allclose(raw[0, 999], crops[1])
