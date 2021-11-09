import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, shape, origin, aggregation_mode='mean', gather_pad=0, gather_fill=0, ):

        self.gather = gather
        self.gather_shape = gather.shape
        self.shape = shape  # rename attributes to crop_shape? self.shape = self.crops.shape
        self.aggregation_mode = aggregation_mode
        self.gather_pad = gather_pad
        self.gather_fill = gather_fill
        self.grid_origins = None
        self.data = self.gather.data
        self.origin = origin  # save origins in np.array
        self.crops = self.make_crops(self.data)


    def make_crops(self, data):
        crops = np.full(shape=(self.origin.shape[0], *self.shape), fill_value=np.nan, dtype=float)  # may be change to np.empty

        for i in range(self.origin.shape[0]):
            crops[i] = self.make_single_crop(self.origin[i], data)
        return crops

    def make_single_crop(self, origin, data):
        shape_y, shape_x = self.gather_shape
        start_y, start_x = origin
        dy, dx = self.shape
        if start_x + dx > shape_x or start_y + dy > shape_y:  # if crop window outs from gather
            result = data[start_y:min(start_y + dy, shape_y),
                          start_x:min(start_x + dx, shape_x)]
            result = np.pad(result, ((0, max(0, min(start_y, shape_y) + dy - shape_y)),
                                     (0, max(0, min(start_x, shape_x) + dx - shape_x))))
            return result
        return data[start_y:start_y + dy, start_x:start_x + dx]

    @batch_method(target='for')
    def assemble_gather(self, input_data=None):
        self.gather.data = None
        gather = self.gather.copy()
        assembling_data = self._assembling(self.crops if input_data is None else input_data)

        gather.data = assembling_data
        return gather

    def _assembling(self, data):
        result = np.zeros(shape=self.gather_shape, dtype=float)
        mask = np.zeros(shape=self.gather_shape, dtype=int)
        for i, origin in enumerate(self.origin):
            result[origin[0]:origin[0] + self.shape[0], origin[1]:origin[1] + self.shape[1]] += data[i]
            mask[origin[0]:origin[0] + self.shape[0], origin[1]:origin[1] + self.shape[1]] += 1
        result = self._aggregate(result, mask, mode=self.aggregation_mode)
        return result

    def _aggregate(self, data, mask, mode):
        if mode == 'mean':
            return data / mask
        else:
            raise NotImplementedError('Using mode are not implement now.')
