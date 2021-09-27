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
        crop_padding=0,
        crop_fill=0,
        cropped_data=[]

    ):
        self.parent = parent_gazer  # настолько ли нам нужен этот газер?
        self.parent_shape = parent_gazer.data.shape
        self.origin = self.make_origin(crop_rule)
        self.crop_size = crop_size
        self.gather_pad = gather_pad
        self.gather_fill = gather_fill
        self.crop_padding = crop_padding
        self.crop_fill = crop_fill
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
        print('start make_crops()')
        crops = []
        data = self.load_data()
        for coords in self.origin:
            crops.append(self.make_single_crop(coords, data))
        return crops
    

    def make_single_crop(self, origin, data):
        print('start make_single_crop()')
        shapes = self.parent_shape
        crop_size = self.crop_size
        # print('crop_padding:', self.crop_padding)

        # move to the separate func
        if self.crop_padding and isinstance(self.crop_padding, list):
            print('maybe this way')
            padding = [self.crop_padding if len(self.crop_padding) >= 2 else [self.crop_padding[0], self.crop_padding[0]]]
        elif self.crop_padding and isinstance(self.crop_padding, int):
            print('this way')
            padding = [self.crop_padding, self.crop_padding]
        elif self.crop_padding and isinstance(self.crop_padding, tuple):
            print('tuple way')
            padding = [tuple(self.crop_padding) if len(self.crop_padding) >= 2 else [self.crop_padding[0], self.crop_padding[0]]][0]
        else:
            padding = [self.crop_padding]
        fill_value = self.crop_fill

        print(f'origin: {origin}, padding: {padding}, crop_size: {crop_size}, shapes: {shapes}')

        start_x, start_y = origin[0] + padding[0], origin[1] + padding[-1]
        dx, dy = crop_size[0] - 2*padding[0], crop_size[1] - 2*padding[-1]
        if start_x + dx > shapes[1]:
            start_x = shapes[1] - dx
        if start_y + dy > shapes[0]:
            start_y = shapes[0] - dy

        print('Cutting shape is', (start_y, start_x, start_y+dy, start_x+dx))
        if self.crop_padding:
            return np.pad(data[start_y:start_y+dy, start_x:start_x+dx], padding, constant_values=self.crop_fill)
        else:
            return data[start_y:start_y+dy, start_x:start_x+dx]


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
            origin = self.origins_from_str(crop_rule)
        return origin


    def origins_from_str(self, crop_rule):
        if crop_rule == 'random':
            return 

    



