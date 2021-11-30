import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, origins, crop_shape, pad_mode='constant', **kwargs):
        self.gather = gather
        self.crop_shape = crop_shape
        self.origins = origins
        self.crops = self.make_crops(self._padding(pad_mode, **kwargs))

    @property
    def n_origins(self):
        return self.origins.shape[0]

    def make_crops(self, data):
        ''' TODO: docs '''
        crops = np.empty(shape=(self.n_origins, *self.crop_shape), dtype=float)
        dx, dy = self.crop_shape
        for i, (start_x, start_y) in enumerate(self.origins):
            crops[i] = data[start_x:start_x + dx, start_y:start_y + dy]
        return crops

    def _padding(self, pad_mode, **kwargs):
        '''Checking if crop window is out of the gather and pad gather to make crop possible. '''
        max_shapes = self.origins.max(axis=0)
        pad_shape_x, pad_shape_y = np.maximum(0, (max_shapes + self.crop_shape - self.gather.shape))
        return np.pad(self.gather.data, ((0, pad_shape_x), (0, pad_shape_y)), mode=pad_mode, **kwargs)

    @batch_method(target='for')
    def assemble_gather(self):
        ''' TODO: docs '''
        assembling_data = self._assembling(self.crops)
        gather = self.gather.copy(ignore='data')
        gather.data = assembling_data
        return gather

    def _assembling(self, data):
        ''' TODO: docs ''' 
        result = np.zeros(shape=self.gather.shape, dtype=float)
        mask = np.zeros(shape=self.gather.shape, dtype=int)
        for crop, origin in zip(data, self.origins):
            result[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]] += crop
            mask[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]] += 1
        return result / mask
