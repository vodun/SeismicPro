import numpy as np

class CroppedGazer:
    ''' class contain '''
    def __init__(
        self,
        parent_gazer,
        crop_rule,
        crop_size,
        gather_pad=0,
        gather_fill=0,
        crop_pad=0,
        crop_fill=0,
        cropped_data=[]
    ):
    
        self.parent = parent_gazer  # настолько ли нам нужен этот газер?
        self.parent_shape = parent_gazer.data.shape
        self.crop_size = crop_size
        self.gather_pad = gather_pad
        self.gather_fill = gather_fill
        self.crop_pad = crop_pad
        self.crop_fill = crop_fill
        self.grid_origins = None
        self.origin = self.make_origin(crop_rule)
        self.crops = self.make_crops()

    def get_croped_data(self):
        print('start get_croped_data()')
        cropped_data = []
        data = load_data(self)
        for item in self.origin:
            cropped_data.append(self.make_single_crop(item))
        setattr(self, 'cropped_data', cropped_data)
        return self
        # or
        # return get_croped_data
    

    def make_crops(self):
        # two ways: save to list or save to numpy array
        # using list now
        print('start make_crops()')
        crops = []
        data = self.load_data()
        coords = np.array(self.origin, dtype=int).reshape(-1, 2)
        print('make_crops, origins', coords)
        for i in range(coords.shape[0]):
            crops.append(self.make_single_crop(coords[i], data))
        return crops


    def make_single_crop(self, origin, data):
        print('start make_single_crop()')
        shapes = self.parent_shape
        crop_size = self.crop_size
        tuple_crop_pad = self.to_tuple(self.crop_pad)

        print(f'origin: {origin}, padding: {self.crop_pad}, crop_pad: {crop_pad}, crop_size: {crop_size}, shapes: {shapes}')

        start_x, start_y = origin[1] + tuple_crop_pad[1][0], origin[0] + tuple_crop_pad[0][0]
        dx, dy = crop_size[1] - sum(tuple_crop_pad[1]), crop_size[0] - sum(tuple_crop_pad[0])
        print(start_x, dx)
        if start_x + dx > shapes[1]:
            start_x = shapes[1] - dx
        if start_y + dy > shapes[0]:
            start_y = shapes[0] - dy

        print('Cutting shape is', (start_y, start_x, start_y+dy, start_x+dx))
        if self.crop_pad:
            return np.pad(data[start_y:start_y+dy, start_x:start_x+dx], tuple_crop_pad, constant_values=self.crop_fill)
        else:
            return data[start_y:start_y+dy, start_x:start_x+dx]

    def assembly_gather(self):
        # do not support a padding
        result = np.zeros(shape=self.parent_shape)
        mask = np.zeros(shape=self.parent_shape) # change to np.zeros !!!
        for origin, crop in zip(self.origin, self.crops):
            # move this logic block to origins_from_str() block
            if origin[0] + self.crop_size[0] > self.parent_shape[0]:
                origin[0] = self.parent_shape[0] - self.crop_size[0]
            if origin[1] + self.crop_size[1] > self.parent_shape[1]:
                origin[1] = self.parent_shape[1] - self.crop_size[1]
            result[origin[0]:origin[0]+self.crop_size[0], origin[1]:origin[1]+self.crop_size[1]] += crop
            mask[origin[0]:origin[0]+self.crop_size[0], origin[1]:origin[1]+self.crop_size[1]] += 1
        result = result / mask
        gather = self.parent.copy()
        gather.data = result
        return gather


    def get_origin_pad(self, crop_pad):
        # нужна ли вообще эта функция?
        ''' return top-left point cropped data based on padding values'''
        origin_pad = ((crop_pad[0][0], crop_pad[0][1]), (crop_pad[1][0], crop_pad[1][1]))
        return origin_pad


    def load_data(self, fill_value=0):
        print('start load_data()')
        if self.gather_pad:
            gather_data = np.pad(self.parent.data, self.gather_pad, constant_values=self.gather_fill)
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
        return np.array(origin, dtype=int).squeeze()


    def origins_from_str(self, crop_rule):
        print('start origins_from_str()')
        if crop_rule == 'random':  # from uniform distribution. Return 
            return (np.random.randint(self.parent_shape[0] + 2 * self.gather_pad[0] - self.crop_size[0]), 
                    np.random.randint(self.parent_shape[1] + 2 * self.gather_pad[1] - self.crop_size[1]))
        elif crop_rule == 'grid':
            print('x_range', 0, self.parent_shape[0], self.crop_size[0])
            origin_x = np.arange(0, self.parent_shape[0], self.crop_size[0])
            print('x_range', 0, self.parent_shape[1], self.crop_size[1])
            origin_y = np.arange(0, self.parent_shape[1], self.crop_size[1])
            return np.array(np.meshgrid(origin_x, origin_y)).T.reshape(-1, 2)
        else:
            raise ValueError('Unknown crop_rule value')


    def to_tuple(self, item):
        if isinstance(item, int):
            return ((item, item), (item, item))
        elif isinstance(item[0], int):
            return ((item[0], item[0]), (item[1], item[1]))
        elif isinstance(item[0], tuple) and isinstance(item[1], tuple):
            return item
        else:
            raise ValueError('Unknown padding value')
    

            
            

