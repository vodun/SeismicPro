import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(self, gather, shape, origins, aggregation_mode='mean', pad_mode='constant'):

        self.gather = gather
        self.crop_shape = shape
        self.aggregation_mode = aggregation_mode
        self.pad_mode = pad_mode
        self.origins = origins
        self.data = self._padding(self.gather.data)
        # self.data = self.gather.data
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
        return data[start_y:start_y + dy, start_x:start_x + dx]

    def _padding(self, data):
        '''Checking if crop window is out of the gather and pad gather to get crops. '''
        pad_mode = self.pad_mode
        shape_y, shape_x = self.gather.shape
        max_start_y, max_start_x = self.origins.max(axis=0)
        dy, dx = self.crop_shape
        pad_shape_x = max(0, max_start_x + dx - shape_x)
        pad_shape_y = max(0, max_start_y + dy - shape_y)
        pad_shape = ((0, pad_shape_y), (0, pad_shape_x))
        if pad_shape[0][1] or pad_shape[1][1]:
            data = np.pad(data, pad_shape, mode=pad_mode, constant_values=0)
        return data

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

    def _assembling(self, data, agg_matrix_raise=1, **kwargs):
        ''' TODO: docs ''' 
        strategy = kwargs.get('strategy', 'greed')
        result = np.full(shape=(*self.gather.shape, agg_matrix_raise), fill_value=np.nan, dtype=float)
        mask = np.full(shape=self.gather.shape, fill_value=-1, dtype=int)

        mask_add = 1

        # print(self.origins)
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
                            # print('index origin:', i)
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
