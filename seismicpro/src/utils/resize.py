""" Stores resizing methods. """

from functools import partial

import cv2
import PIL
from scipy import signal
import numpy as np


def sinc_resize(image, new_shape):
    """ functional form for pillow sinc resize. """
    img = PIL.Image.fromarray(image, mode='F')
    new_img = img.resize(new_shape, resample=PIL.Image.LANCZOS)
    return np.array(new_img)

def fft_resize(image, new_shape):
    """ functional form for scipy fft resize. """
    return signal.resample(image, new_shape[0], axis=1)


INTERPOLATORS = {
    'nearest' : partial(cv2.resize, interpolation=cv2.INTER_NEAREST),
    'linear' : partial(cv2.resize, interpolation=cv2.INTER_LINEAR),
    'cubic' : partial(cv2.resize, interpolation=cv2.INTER_CUBIC),
    'sinc' : sinc_resize,
    'fft' : fft_resize
}
