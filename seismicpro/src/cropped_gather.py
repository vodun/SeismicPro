import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, crop_shape, origins, pad_kwargs={}):
        self.gather = gather
        self.crop_shape = crop_shape
        self.n_origins = origins.shape[0]
        self.origins = origins
        self.crops = self.make_crops(self._padding(pad_kwargs))

    def make_crops(self, data):
        ''' TODO: docs '''
        crops = np.empty(shape=(self.n_origins, *self.crop_shape), dtype=float)
        dx, dy = self.crop_shape
        for i, (start_x, start_y) in enumerate(self.origins):
            crops[i] = data[start_x:start_x + dx, start_y:start_y + dy]
        return crops

    def _padding(self, pad_kwargs):
        '''Checking if crop window is out of the gather and pad gather to get crops. '''
        max_shapes = np.atleast_2d(self.origins.max(axis=0))
        temp_max = (max_shapes + self.crop_shape - self.gather.shape).max(axis=0)
        pad_shape_x, pad_shape_y = np.maximum(np.zeros(shape=2, dtype=int), temp_max)
        return np.pad(self.gather.data, ((0, pad_shape_x), (0, pad_shape_y)), **pad_kwargs)

    @batch_method(target='for')
    def assemble_gather(self, aggregation_mode='mean'):
        ''' TODO: docs '''
        # test save_to in predict_model
        assembling_data = self._assembling(self.crops, aggregation_mode=aggregation_mode)
        # avoiding gather data copying 
        rest_data = self.gather.data
        self.gather.data = None
        gather = self.gather.copy()
        self.gather.data = rest_data

        gather.data = assembling_data
        return gather

    def _assembling(self, data, aggregation_mode):
        ''' TODO: docs ''' 
        result = np.zeros(shape=self.gather.shape, dtype=float)
        mask = np.zeros(shape=self.gather.shape, dtype=int)
        for i, origin in enumerate(self.origins):
            result[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]] += data[i]
            mask[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]] += 1
        result = self._aggregate(result, mask, mode=aggregation_mode)
        return result

    def _aggregate(self, data, mask, mode):
        ''' TODO: docs '''
        if mode == 'mean':
            return data / mask
        else:
            raise NotImplementedError('Using mode are not implement now.')
