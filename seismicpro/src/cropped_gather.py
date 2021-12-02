import warnings

import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, origins, crop_shape, pad_mode='constant', **kwargs):
        self.gather = gather
        self.crop_shape = crop_shape
        self.origins = origins
        self.crops = self.make_crops(self._gather_pad(pad_mode, **kwargs))

    @property
    def n_origins(self):
        return self.origins.shape[0]

    def make_crops(self, data):
        ''' TODO: docs '''
        crops = np.empty(shape=(self.n_origins, *self.crop_shape), dtype=data.dtype)
        dx, dy = self.crop_shape
        for i, (start_x, start_y) in enumerate(self.origins):
            crops[i] = data[start_x:start_x + dx, start_y:start_y + dy]
        return crops

    def _gather_pad(self, pad_mode, **kwargs):
        '''Checking if crop window is out of the gather and pad gather to make crop possible. '''
        max_shapes = self.origins.max(axis=0)
        pad_shape_x, pad_shape_y = np.maximum(0, max_shapes + self.crop_shape - self.gather.shape)
        if (pad_shape_x > 0) or (pad_shape_y > 0):
            warnings.warn("Crop is out of the gather data. The Gather's data will be padded")
            return np.pad(self.gather.data, ((0, pad_shape_x), (0, pad_shape_y)), mode=pad_mode, **kwargs)
        return self.gather.data

    @batch_method(target='for')
    def assemble_gather(self):
        ''' TODO: docs '''
        assembling_data = self._mean_assemble()
        gather = self.gather.copy(ignore='data')
        gather.data = assembling_data
        return gather

    def _mean_assemble(self):
        ''' TODO: docs ''' 
        result_shape = np.maximum(self.gather.shape, self.crop_shape + self.origins.max(axis=0))
        result = np.zeros(shape=result_shape, dtype=float)
        mask = np.zeros(shape=result_shape, dtype=int)
        for crop, origin in zip(self.crops, self.origins):
            result[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]] += crop
            mask[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]] += 1
        result /= mask
        return result[:self.gather.shape[0], :self.gather.shape[1]]
