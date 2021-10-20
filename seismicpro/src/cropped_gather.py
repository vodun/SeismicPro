import numpy as np

from .decorators import batch_method

class CroppedGather:
    ''' class contain '''
    def __init__(
        self,
        parent_gather,
        crop_rule,
        crop_size,
        gather_pad=0,
        gather_fill=0,
        cropped_data=None
    ):
    
        self.parent = parent_gather
        self.parent_shape = parent_gather.data.shape
        self.crop_size = crop_size
        self.gather_pad = gather_pad
        self.gather_fill = gather_fill
        self.grid_origins = None
        self.origin = self.make_origin(crop_rule)  # save origins in np.array only
        self.crops = self.make_crops() # self.make_crops()
        # if hasattr(self.parent, 'mask'):
        #     self.mask_crops = self.make_crops('mask')


    def make_crops(self):
        # two ways: save to list or save to numpy array
        # using numpy array now
        # make_model_inputs() ?
        print('start make_crops()')

        data = self.load_data()
        crops = np.full(shape=(len(self.origin), *self.crop_size), fill_value=np.nan, dtype=float)
        print('origins', self.origin.shape)
        print('crop shape', crops.shape)
        coords = np.array(self.origin, dtype=int).reshape(-1, 2)  # move to make_origin
        print('make_crops, origins', coords)
        for i in range(coords.shape[0]):
            crops[i, :, :] = self.make_single_crop(coords[i], data) # np.array way
            # crops.append(self.make_single_crop(coords[i], data))  # list way
            print('iter crops shape', crops.shape)
        return crops


    def make_single_crop(self, origin, data):
        print('start make_single_crop()')
        shapes = self.parent_shape
        crop_size = self.crop_size

        # print(f'origin: {origin}, padding: {self.crop_pad}, crop_pad: {tuple_crop_pad}, crop_size: {crop_size}, shapes: {shapes}')

        start_x, start_y = origin[1], origin[0]
        dx, dy = crop_size[1], crop_size[0]
        # print(start_x, dx)
        if start_x + dx > shapes[1] or start_y + dy > shapes[0]: # if crop window outs from gather
            result = data[start_y:min(start_y+dy, start_y + self.crop_size[1]), start_x:min(start_x+dx, start_x + self.crop_size[0])]
            result = np.pad(result, ((0, max(0, min(start_y, shapes[0]) + dy - self.parent_shape[0])), 
                                     (0, max(0, min(start_x, shapes[1]) + dx - self.parent_shape[1]))) )
            return result
        return data[start_y:start_y+dy, start_x:start_x+dx]


    @batch_method(target='for')
    def assembly_gather(self):
        print('start assembly_gather()')
        result = np.zeros(shape=self.parent_shape, dtype=float)
        mask = np.zeros(shape=self.parent_shape, dtype=int)
        for i, origin in enumerate(self.origin):
            result[origin[0]:origin[0]+self.crop_size[0], origin[1]:origin[1]+self.crop_size[1]] = \
                result[origin[0]:origin[0]+self.crop_size[0], origin[1]:origin[1]+self.crop_size[1]] + self.crops[i, :, :]
            # np.add(result[origin[0]:origin[0]+self.crop_size[0], origin[1]:origin[1]+self.crop_size[1]], crop, out=result)
            mask[origin[0]:origin[0]+self.crop_size[0], origin[1]:origin[1]+self.crop_size[1]] += 1
        result /= mask
        gather = self.parent.copy()
        gather.data = result
        return gather


    def load_data(self, fill_value=0):
        print('start load_data()')
        if self.gather_pad:
            gather_data = np.pad(self.parent.data, self.to_tuple(self.gather_pad), constant_values=self.gather_fill)
        else: 
            gather_data = self.parent.data
        return gather_data


    def make_origin(self, crop_rule):
        print('start make_origin()')
        origin = []
        if isinstance(crop_rule, tuple):
            origin.append(crop_rule)
        elif isinstance(crop_rule, int):
            origin.append(tuple(crop_rule, crop_rule))
        elif isinstance(crop_rule, list):
            origin = crop_rule
        elif isinstance(crop_rule, str):
            origin.append(self.origins_from_str(crop_rule))
        return np.array(origin, dtype=int)


    def origins_from_str(self, crop_rule):
        print('start origins_from_str()')
        print(self.gather_pad, self.parent_shape)
        if crop_rule == 'random':  # from uniform distribution. 
            # issue: return one point only
            return (np.random.randint(self.parent_shape[0] - self.crop_size[0]), 
                    np.random.randint(self.parent_shape[1] - self.crop_size[1]))
        elif crop_rule == 'grid': # do not support padding
            print('x_range', 0, self.parent_shape[0], self.crop_size[0])
            origin_x = np.arange(0, self.parent_shape[0], self.crop_size[0], dtype=int)
            print('y_range', 0, self.parent_shape[1], self.crop_size[1])
            origin_y = np.arange(0, self.parent_shape[1], self.crop_size[1], dtype=int)
            # correct origin logic should be confirmed
            # is drop last is needed            
            if origin_x[-1] + self.crop_size[0] > self.parent_shape[0]:
                origin_x[-1] = self.parent_shape[0] - self.crop_size[0]
            if origin_y[-1] + self.crop_size[1] > self.parent_shape[1]:
                origin_y[-1] = self.parent_shape[1] - self.crop_size[1]
            return np.array(np.meshgrid(origin_x, origin_y)).T.reshape(-1, 2)
        else:
            raise ValueError('Unknown crop_rule value')


    def to_tuple(self, item):  # maybe remove
        if isinstance(item, int):
            return ((item, item), (item, item))
        elif isinstance(item[0], int):
            return ((item[0], item[0]), (item[1], item[1]))
        elif isinstance(item[0], tuple) and isinstance(item[1], tuple):
            return item
        else:
            raise ValueError('Unknown padding value')
