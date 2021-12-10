import warnings

import numpy as np

from .decorators import batch_method


class CroppedGather:
    """A class represented cropped data of a Gather class.
    
    CroppedGather instance can be created by `crop` method of a Gather class. Crops are calculated automatically and 
    kept in the `crops` attribute.
    Two main ways to use CroppedGather:
        training a 2d ML model: 
            1. random crop a gathers with `crop` method of a Gather class.
            2. use crops as a training dataset for your model.
        inference a 2d ML models: 
            1. crop full gather data with origins='grid' in crop method of a Gather class.
            2. use crops as a test dataset for your model.
            3. save the prediction of your model to `crops` attribute.
            4. assemble gather from predicted crops with assemble_gather method of a CroppedGather.
        
    Examples
    --------
    Create a Croppedgather instance and get single crop from origin (0, 0) and crop shape is (100, 100):
    >>> croppedgather = gather.crop(origins=(0, 0), crop_shape=(100, 100))
    Create a Croppedgather and get 10 randomly selected crops with shape of each crop is (100, 100):
    >>> croppedgather = gather.crop(origins='random', crop_shape=(100, 100), n_items=10)

    Parameters
    ----------
    gather : Gather
        Gather to crop.
    origins : np.ndarray
        2d NumPy array with x and y coordinate of each origin, where x corresponds with samples and y corresponds with
        traces.
    crop_shape: tuple
        Shape of each crop.
    pad_mode : str
        mode of `np.pad` function to pad `gather` data.
    kwargs : misc, optional
        Additional keyword arguments for padding `gather` data.
    
    Attributes
    ----------
    gather : Gather
        Gather to crop.
    origins : np.ndarray
        2d NumPy array with x and y coordinate of each origin, where x corresponds with samples and y corresponds with
        traces.
    crop_shape: tuple
        Shape of each crop.
    crops : np.ndarray
        All crops are kept in one 3d np.ndarray. Shape of array is (n_crops, *(crop_shape))
    """
    def __init__(self, gather, origins, crop_shape, pad_mode, **kwargs):
        self.gather = gather
        self.origins = origins
        self.crop_shape = crop_shape
        self.crops = self._make_crops(self._pad_gather(mode=pad_mode, **kwargs))

    @property
    def n_origins(self):
        '''Number of origins in the `origins` attribute.'''
        return self.origins.shape[0]

    def _make_crops(self, data):
        '''Crop the given data.
        
        Parameters
        ----------
        data : np.ndarray
            Gather's data kept in 2d array.
        
        Returns
        -------
        crops : np.ndarray
            All crops are kept in one 3d array.
        '''
        crops = np.empty(shape=(self.n_origins, *self.crop_shape), dtype=data.dtype)
        dx, dy = self.crop_shape
        for i, (x_0, y_0) in enumerate(self.origins):
            crops[i] = data[x_0:x_0 + dx, y_0:y_0 + dy]
        return crops

    def _pad_gather(self, **kwargs):
        '''Pad the `gather` data if it needs to makecrop. 
        
        Parameters
        ----------
        kwargs : misc, optional
            Keywords to `np.pad` function.
        
        Returns
        -------
        data : np.ndarray
            Gather data with needful padding.

        Warnings
        --------
        Shows if padding is needed.
        '''
        max_origins = self.origins.max(axis=0)
        pad_width_x, pad_width_y = np.maximum(0, max_origins + self.crop_shape - self.gather.shape)
        if (pad_width_x > 0) or (pad_width_y > 0):
            warnings.warn("Crop is out of the gather data. The Gather's data will be padded")
            return np.pad(self.gather.data, ((0, pad_width_x), (0, pad_width_y)), **kwargs)
        return self.gather.data

    @batch_method(target='for', copy_src=False)
    def assemble_gather(self):
        '''Assemble gather from crops.
        
        `assemble_gather` uses crops and origins to assemble gather data. The resulting gather will be identically 
        with gather used to create CroppedGather instance except for data attribute. If no crops the data corresponds
        to some data array element then np.nan will keep in this cell. Use `origins='grid'` as `crop`'s method 
        parameters to avoid this.

        Returns
        -------
        gather : Gather
            Gather data assembled from crops. Other gather's attributes will copy from gather used to create 
            CroppedGather instance.
        '''
        assembled_data = self._assemble_mean()
        gather = self.gather.copy(ignore='data')
        gather.data = assembled_data
        return gather

    def _assemble_mean(self):
        '''Assemble and mean aggregate crops.

        Returns
        -------
        crops_sum : np.ndarray
            2d array with a point-by-point sum of all crops normalized at a number of crops cover.
        ''' 
        padded_gather_shape = np.maximum(self.gather.shape, self.crop_shape + self.origins.max(axis=0))
        crops_sum = np.zeros(shape=padded_gather_shape, dtype=np.float32)
        crops_count = np.zeros(shape=padded_gather_shape, dtype=np.int16)
        dx, dy = self.crop_shape
        for crop, (x_0, y_0) in zip(self.crops, self.origins):
            crops_sum[x_0:x_0 + dx, y_0:y_0 + dy] += crop
            crops_count[x_0:x_0 + dx, y_0:y_0 + dy] += 1
        crops_sum /= crops_count
        return crops_sum[:self.gather.shape[0], :self.gather.shape[1]]
