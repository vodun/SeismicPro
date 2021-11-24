import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, shape, origins, aggregation_mode='mean', pad_mode='constant'):
        self.gather = gather
        self.crop_shape = shape
        self.aggregation_mode = aggregation_mode
        self.origins = origins
        self.crops = self.make_crops(self._padding(pad_mode))

    def make_crops(self, data):
        ''' TODO: docs '''
        crops = np.empty(shape=(self.origins.shape[0], *self.crop_shape), dtype=float)

        dx, dy = self.crop_shape
        for i, (start_x, start_y) in enumerate(self.origins):
            crops[i] = data[start_x:start_x + dx, start_y:start_y + dy]
        return crops

    def _padding(self, pad_mode):
        '''Checking if crop window is out of the gather and pad gather to get crops. '''
        max_shapes = np.atleast_2d(self.origins.max(axis=0))
        temp_max = (max_shapes + self.crop_shape - self.gather.shape).max(axis=0)
        pad_shape_x, pad_shape_y = np.maximum(np.zeros(shape=2, dtype=int), temp_max)
        return np.pad(self.gather.data, ((0, pad_shape_x), (0, pad_shape_y)), mode=pad_mode)

    @batch_method(target='for')
    def assemble_gather(self, input_data=None, **kwargs):
        ''' TODO: docs '''
        # test save_to in predict_model
        assembling_data = self._assembling(self.crops if input_data is None else input_data, **kwargs)

        # avoiding gather data copying 
        rest_data = self.gather.data
        self.gather.data = None
        gather = self.gather.copy()
        self.gather.data = rest_data

        gather.data = assembling_data
        return gather

    def _assembling(self, data, agg_matrix_raise=1, strategy='greed'):
        ''' TODO: docs ''' 
        result = np.full(shape=(*self.gather.shape, agg_matrix_raise), fill_value=np.nan, dtype=float)
        mask = np.full(shape=self.gather.shape, fill_value=-1, dtype=int)

        mask_add = 1
        if strategy == 'rand':
            n_origins = self.origins.shape[0]
            rand_i = np.random.permutation(np.arange(n_origins))
            for i in rand_i:
                one_crop = data[i]
                origin = self.origins[i]
                # index where I want to write next crop
                plate_index = mask[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]].max() + 1
                # checking if result array need to extend by axis=2
                if plate_index >= result.shape[2]:
                    result = np.dstack((result, np.full(shape=(*self.gather.shape, agg_matrix_raise), fill_value=np.nan, dtype=float)))
                result[origin[0]:origin[0] + self.crop_shape[0], origin[1]:(origin[1] + self.crop_shape[1]), plate_index] = one_crop
                mask[origin[0]:origin[0] + self.crop_shape[0], origin[1]:origin[1] + self.crop_shape[1]] = plate_index
            print('Rand aggregate matrix shape:', result.shape)

        # based on origins structure
        # greed strategy at 3 times faster and aggregate matrix(result) 2 times smaller
        if strategy == 'greed':
            masked_origins = np.ma.array(self.origins, mask=False)
            level = 0
            while masked_origins.mask.sum() < self.origins.size:
                border_x, border_y = 0, 0
                for origin in masked_origins:
                    if (origin.mask == False).all():
                        if origin[0] >= border_x or origin[1] >= border_y:
                            # getting index of origin to get crop
                            i, = np.where(np.all(self.origins == origin, axis=1))
                            one_crop = data[i]
                            result[origin[0]:origin[0] + self.crop_shape[0], origin[1]:(origin[1] + self.crop_shape[1]), level] = one_crop
                            border_x = origin[0] + self.crop_shape[0]
                            border_y = origin[1] + self.crop_shape[1]
                            masked_origins.mask[i] = True
                result = np.dstack((result, np.full(shape=(*self.gather.shape, 1), fill_value=np.nan, dtype=float)))
                level += 1
            print('Greed aggregate matrix shape:', result.shape)


        result = self._aggregate(result, mode=self.aggregation_mode)
        return result

    def _aggregate(self, data, mode):
        ''' TODO: docs '''
        if mode == 'mean':
            return np.nanmean(data, axis=2)
        else:
            raise NotImplementedError('Using mode are not implement now.')
