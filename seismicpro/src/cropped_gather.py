import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, shape, origins, aggregation_mode='mean'):

        self.gather = gather
        self.crop_shape = shape
        self.aggregation_mode = aggregation_mode
        self.data = self.gather.data
        self.origins = origins
        self.crops = self.make_crops(self.data)

    def make_crops(self, data):
        ''' TODO: docs '''
        crops = np.full(shape=(self.origins.shape[0], *self.crop_shape), fill_value=np.nan, dtype=float)  # may be change to np.empty

        for i in range(self.origins.shape[0]):
            crops[i] = self.make_single_crop(self.origins[i], data)
        return crops

    def make_single_crop(self, origin, data):
        ''' TODO: docs '''
        shape_y, shape_x = self.gather.shape
        start_y, start_x = origin
        dy, dx = self.crop_shape
        if start_x + dx > shape_x or start_y + dy > shape_y:  # if crop window outs from gather
            result = data[start_y:min(start_y + dy, shape_y),
                          start_x:min(start_x + dx, shape_x)]
            result = np.pad(result, ((0, max(0, min(start_y, shape_y) + dy - shape_y)),
                                     (0, max(0, min(start_x, shape_x) + dx - shape_x))))
            return result
        return data[start_y:start_y + dy, start_x:start_x + dx]

    @batch_method(target='for')
    def assemble_gather(self, input_data=None, **kwargs):
        ''' TODO: docs '''
        assembling_data = self._assembling(self.crops if input_data is None else input_data, **kwargs)

        # avoiding gather data copying 
        rest_data = self.gather.data
        self.gather.data = None
        gather = self.gather.copy()
        self.gather.data = rest_data

        gather.data = assembling_data
        return gather

    def _assembling(self, data, **kwargs):
        ''' TODO: docs ''' 
        result = np.zeros(shape=self.gather.shape, dtype=float)
        mask = np.zeros(shape=self.gather.shape, dtype=int)

        cut_value = kwargs.get('grid_cut_value', 0)
        cut_edge = kwargs.get('grid_cut_edge', (0, 0))

        mask_add = np.pad(np.ones(shape=(self.crop_shape[0] - 2 * cut_edge[0], 
                                         self.crop_shape[1] - 2 * cut_edge[1]), dtype=int), 
                                  pad_width=((cut_edge[0], cut_edge[0]), (cut_edge[1], cut_edge[1])),
                                  constant_values=0)

        for i, origin in enumerate(self.origins):
            one_crop = data[i]
            # padding edge with zero. no shapes changes.
            one_crop = np.pad(one_crop[cut_edge[0]:(self.crop_shape[0] - cut_edge[0]), 
                                       cut_edge[1]:(self.crop_shape[1] - cut_edge[1])], 
                                pad_width=((cut_edge[0], cut_edge[0]), (cut_edge[1], cut_edge[1])),
                                constant_values=cut_value)

            result[origin[0]:origin[0] + self.crop_shape[0], origin[1]:(origin[1] + self.crop_shape[1])] += one_crop
            mask[origin[0]:origin[0] + self.crop_shape[0], origin[1]:(origin[1] + self.crop_shape[1])] += mask_add
        result = self._aggregate(result, mask, mode=self.aggregation_mode)
        return result

    def _aggregate(self, data, mask, mode):
        ''' TODO: docs '''
        if mode == 'mean':
            return data / mask
        else:
            raise NotImplementedError('Using mode are not implement now.')
