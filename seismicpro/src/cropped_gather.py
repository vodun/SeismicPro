import numpy as np

from .decorators import batch_method


class CroppedGather:
    ''' cool docstring here '''

    def __init__(
            self,
            parent_gather,
            shape,
            origin,
            gather_pad=0,
            gather_fill=0,
            is_mask=False
    ):

        self.parent = parent_gather
        self.parent_shape = parent_gather.data.shape
        self.shape = shape
        self.gather_pad = gather_pad
        self.gather_fill = gather_fill
        self.grid_origins = None
        self.data = self.load_data()
        self.origin = origin  # save origins in np.array only
        self.crops = self.make_crops(self.data)  # self.make_crops()
        self.is_mask = is_mask

        if self.is_mask:  # two way. crop mask automatical or use key 'crop_mask'
            if hasattr(self.parent, 'mask'):
                self.crops_mask = self.make_crops(self.parent.mask)
            else:
                raise AttributeError("Gather hasn't a mask to crop.")

    def make_crops(self, data):
        # two ways: save to list or save to numpy array
        # using numpy array now
        # make_model_inputs() ?
        # print('start make_crops()')
        crops = np.full(shape=(self.origin.shape[0], *self.shape), fill_value=np.nan, dtype=float)

        for i in range(self.origin.shape[0]):
            crops[i, :, :] = self.make_single_crop(self.origin[i], data)
            # print('iter crops shape', crops.shape)
        return crops

    def make_single_crop(self, origin, data):
        # print('start make_single_crop()')
        shapes = self.parent_shape
        start_x, start_y = origin[1], origin[0]
        dx, dy = self.shape[1], self.shape[0]
        # print(start_x, dx)
        if start_x + dx > self.parent_shape[1] or start_y + dy > self.parent_shape[
            0]:  # if crop window outs from gather
            result = data[start_y:min(start_y + dy, start_y + self.shape[1]),
                     start_x:min(start_x + dx, start_x + self.shape[0])]
            result = np.pad(result, ((0, max(0, min(start_y, self.parent_shape[0]) + dy - self.parent_shape[0])),
                                     (0, max(0, min(start_x, self.parent_shape[1]) + dx - self.parent_shape[1]))))
            return result
        return data[start_y:start_y + dy, start_x:start_x + dx]

    @batch_method(target='for')
    def assemble_gather(self, component='data', input_data=None):
        # print('start assembly_gather()')
        gather = self.parent.copy()

        if component == 'data':
            gather.data = self._assembling(self.crops)
        elif component == 'mask':
            gather.mask = None
            if input_data is None:
                assembling_data = self._assembling(self.crops_mask)
            else:
                assembling_data = self._assembling(input_data)
            setattr(gather, component, assembling_data)
        else:
            raise ValueError('Unknown component.')
        return gather

    def _assembling(self, data):
        # print('start _assembling')
        result = np.zeros(shape=self.parent_shape, dtype=float)
        mask = np.zeros(shape=self.parent_shape, dtype=int)
        # print(data.shape, result.shape, mask.shape)
        for i, origin in enumerate(self.origin):
            result[origin[0]:origin[0] + self.shape[0], origin[1]:origin[1] + self.shape[1]] += data[i, :, :]
            mask[origin[0]:origin[0] + self.shape[0], origin[1]:origin[1] + self.shape[1]] += 1
        result /= mask
        return result

    def load_data(self):
        # print('start load_data()')
        if self.gather_pad:
            gather_data = np.pad(self.parent.data, self.to_tuple(self.gather_pad), constant_values=self.gather_fill)
        else:
            gather_data = self.parent.data
        return gather_data


    def to_tuple(self, item):  # maybe remove
        if isinstance(item, int):
            return ((item, item), (item, item))
        elif isinstance(item[0], int):
            return ((item[0], item[0]), (item[1], item[1]))
        elif isinstance(item[0], tuple) and isinstance(item[1], tuple):
            return item
        else:
            raise ValueError('Unknown padding value')
