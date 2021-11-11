import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, shape, origin, aggregation_mode='mean'):

        self.gather = gather
        self.gather_shape = gather.shape
        self.shape = shape  # rename attributes to crop_shape? self.shape = self.crops.shape
        self.aggregation_mode = aggregation_mode
        self.data = self.gather.data
        self.origin = origin  # save origins in np.array
        self.crops = self.make_crops(self.data)

    def make_crops(self, data):
        ''' TO DO: docs '''
        crops = np.full(shape=(self.origin.shape[0], *self.shape), fill_value=np.nan, dtype=float)  # may be change to np.empty

        for i in range(self.origin.shape[0]):
            crops[i] = self.make_single_crop(self.origin[i], data)
        return crops

    def make_single_crop(self, origin, data):
        ''' TO DO: docs '''
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
    def assemble_gather(self, input_data=None, **kwargs):
        ''' TO DO: docs '''
        assembling_data = self._assembling(self.crops if input_data is None else input_data, **kwargs)

        self.gather.data = None
        gather = self.gather.copy()
        gather.data = assembling_data
        return gather

    def _assembling(self, data, **kwargs):
        ''' TO DO: docs '''
        result = np.zeros(shape=self.gather_shape, dtype=float)
        mask = np.zeros(shape=self.gather_shape, dtype=int)
        for i, origin in enumerate(self.origin):
            one_crop = data[i]
            mask_add = 1
            # edge cutting with zero
            if ('grid_cut_value' in kwargs.keys()) and ('grid_cut_edge' in kwargs.keys()):
                cut_value = kwargs['grid_cut_value']
                cut_edge = kwargs['grid_cut_edge']
                one_crop = np.pad(one_crop[cut_edge[0]:(self.shape[0] - cut_edge[0]), 
                                           cut_edge[1]:(self.shape[1] - cut_edge[1])], 
                                  pad_width=((cut_edge[0], cut_edge[0]), (cut_edge[1], cut_edge[1])),
                                  constant_values=cut_value)
                mask_add = np.pad(np.ones(shape=(self.shape[0] - 2 * cut_edge[0], 
                                                 self.shape[1] - 2 * cut_edge[1]), 
                                          dtype=int), 
                                  pad_width=((cut_edge[0], cut_edge[0]), (cut_edge[1], cut_edge[1])),
                                  constant_values=0)
            result[origin[0]:origin[0] + self.shape[0], origin[1]:(origin[1] + self.shape[1])] += one_crop
            mask[origin[0]:origin[0] + self.shape[0], origin[1]:(origin[1] + self.shape[1])] += mask_add
        result = self._aggregate(result, mask, mode=self.aggregation_mode)
        return result

    def _aggregate(self, data, mask, mode):
        ''' TO DO: docs '''
        if mode == 'mean':
            return data / mask
        else:
            raise NotImplementedError('Using mode are not implement now.')
