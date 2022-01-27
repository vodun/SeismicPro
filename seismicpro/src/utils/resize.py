"""Stores resizing methods. """

from functools import partial

import cv2
import PIL
import scipy


def sinc_resize(image, new_shape, resample):
    img = PIL.Image.fromarray(image, mode='F')
    new_img = img.resize(new_shape, resample=resample)
    return np.array(new_img)

def fft_resize(image, new_shape):
    return signal.resample(image, new_shape[0], axis=1)


INTERPOLATORS = {
    'nearest' : partial(cv2.resize, interpolation=cv2.INTER_NEAREST),
    'linear' : partial(cv2.resize, interpolation=cv2.INTER_LINEAR),
    'cubic' : partial(cv2.resize, interpolation=cv2.INTER_CUBIC),
    'sinc' : partial(sinc_resize, resample=PIL.Image.LANCZOS),
    'fft' : fft_resize
}

