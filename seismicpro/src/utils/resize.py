""" Stores resizing methods. """

from functools import partial

import cv2
import PIL
import numpy as np
import numba
from scipy import signal


def sinc_resize(image, samples, new_samples):
    """ functional form for pillow sinc resize. """
    img = PIL.Image.fromarray(image, mode='F')
    new_img = img.resize((len(new_samples), image.shape[0]), resample=PIL.Image.LANCZOS)
    return np.array(new_img)

def fft_resize(image, samples, new_samples):
    """ functional form for scipy fft resize. """
    return signal.resample(image, len(new_samples), axis=1)

@numba.njit(nogil=True, fastmath=True, parallel=True)
def linear_resize(data, samples, new_samples):
    res = np.empty((len(data), len(new_samples)), dtype=np.float32)
    for i in numba.prange(len(data)):
        res[i] = np.interp(new_samples, samples, data[i])
    return res


INTERPOLATORS = {
    'nearest' : partial(cv2.resize, interpolation=cv2.INTER_NEAREST),
    'linear' : partial(cv2.resize, interpolation=cv2.INTER_LINEAR),
    'cubic' : partial(cv2.resize, interpolation=cv2.INTER_CUBIC),
    'sinc' : sinc_resize,
    'fft' : fft_resize
}
