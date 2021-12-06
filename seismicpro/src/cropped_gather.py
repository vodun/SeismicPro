import warnings

import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, origins, crop_shape, pad_mode='constant', **kwargs):
        self.gather = gather
        self.crop_shape = crop_shape
        self.origins = origins
        self.crops = self.make_crops(self._pad_gather(pad_mode, **kwargs))

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

    def _pad_gather(self, pad_mode, **kwargs):
        '''Checking if crop window is out of the gather and pad gather to make crop possible. '''
        max_origins_distance = self.origins.max(axis=0)
        pad_width_x, pad_width_y = np.maximum(0, max_origins_distance + self.crop_shape - self.gather.shape)
        if (pad_width_x > 0) or (pad_width_y > 0):
            warnings.warn("Crop is out of the gather data. The Gather's data will be padded")
            return np.pad(self.gather.data, ((0, pad_width_x), (0, pad_width_y)), mode=pad_mode, **kwargs)
        return self.gather.data

    @batch_method(target='for')
    def assemble_gather(self):
        ''' TODO: docs '''
        assembled_data = self._assemble_mean()
        gather = self.gather.copy(ignore='data')
        gather.data = assembled_data
        return gather

    def _assemble_mean(self):
        ''' TODO: docs ''' 
        used_gather_shape = np.maximum(self.gather.shape, self.crop_shape + self.origins.max(axis=0))
        agg_crops = np.zeros(shape=used_gather_shape, dtype=np.float32)
        count_crops = np.zeros(shape=used_gather_shape, dtype=int)
        for crop, origin in zip(self.crops, self.origins):
            agg_crops[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]] += crop
            count_crops[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]] += 1
        agg_crops /= count_crops
        return agg_crops[:self.gather.shape[0], :self.gather.shape[1]]
