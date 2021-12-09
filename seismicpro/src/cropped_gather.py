import warnings

import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, origins, crop_shape, **kwargs):
        self.gather = gather
        self.crop_shape = crop_shape
        self.origins = origins
        self.crops = self.make_crops(self._pad_gather(**kwargs))

    @property
    def n_origins(self):
        return self.origins.shape[0]

    def make_crops(self, data):
        ''' TODO: docs '''
        crops = np.empty(shape=(self.n_origins, *self.crop_shape), dtype=data.dtype)
        dx, dy = self.crop_shape
        for i, (x_0, y_0) in enumerate(self.origins):
            crops[i] = data[x_0:x_0 + dx, y_0:y_0 + dy]
        return crops

    def _pad_gather(self, **kwargs):
        '''Checking if crop window is out of the gather and pad gather to make crop possible. '''
        max_origins_distance = self.origins.max(axis=0)
        pad_width_x, pad_width_y = np.maximum(0, max_origins_distance + self.crop_shape - self.gather.shape)
        if (pad_width_x > 0) or (pad_width_y > 0):
            warnings.warn("Crop is out of the gather data. The Gather's data will be padded")
            return np.pad(self.gather.data, ((0, pad_width_x), (0, pad_width_y)), mode=kwargs.pop('pad_mode'), **kwargs)
        return self.gather.data

    @batch_method(target='for', copy_src=False)
    def assemble_gather(self):
        ''' TODO: docs '''
        assembled_data = self._assemble_mean()
        gather = self.gather.copy(ignore='data')
        gather.data = assembled_data
        return gather

    def _assemble_mean(self):
        ''' TODO: docs ''' 
        padded_gather_shape = np.maximum(self.gather.shape, self.crop_shape + self.origins.max(axis=0))
        crops_sum = np.zeros(shape=padded_gather_shape, dtype=np.float32)
        crops_count = np.zeros(shape=padded_gather_shape, dtype=np.int16)
        for crop, (x_0, y_0) in zip(self.crops, self.origins):
            crops_sum[x_0:x_0 + self.crop_shape[0], y_0:y_0 + self.crop_shape[1]] += crop
            crops_count[x_0:x_0 + self.crop_shape[0], y_0:y_0 + self.crop_shape[1]] += 1
        crops_sum /= crops_count
        return crops_sum[:self.gather.shape[0], :self.gather.shape[1]]
