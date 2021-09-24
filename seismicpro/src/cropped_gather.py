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
        print(1)
        self.parent = parent_gazer  # настолько ли нам нужен этот газер?
        self.parent_shape = parent_gazer.data.shape
        self.crop_rule = crop_rule
        self.crop_size = crop_size
        self.gather_pad = gather_pad
        self.gather_fill = gather_fill
        self.crop_padding = crop_padding
        self.crop_fill = crop_fill

    def get_croped_data(self):
        cropped_data = []
        if isinstance(crop_rule, list):
            data = load_data(self)
            for item in crop_rule:
                cropped_data.append(make_single_crop(item))
        setattr(self, 'cropped_data', cropped_data)
        return self
        # or
        # return get_croped_data
    
    def make_single_crop(self, origin):
        shapes = self.parent_shape
        crop_size = self.crop_size
        if not self.crop_padding:
            padding = list(self.crop_padding)
        else:
            padding = self.crop_padding
        fill_value = self.crop_fill

        start_x, start_y = origin[0] + padding[0], origin[1] + padding[-1]
        dx, dy = crop_size[0] - 2*padding[0], crop_size[1] - 2*padding[-1]
        if start_x + dx > shapes[1]:
            start_x = shapes[1] - dx
        if start_y + dy > shapes[0]:
            start_y = shapes[0] - dy
        self.cropped_data = self.data[start_y:start_y+dy, start_x:start_x+dx]

    def load_data(self, fill_value=0):
        if self.gather_pad:
            gather_data = np.pad(self.parent.data, self.gather_pad, constant_values=self.gather_fill)
        else: 
            gather_data = self.parent.data
        return gather_data
    



