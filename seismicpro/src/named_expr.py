import sys

import numpy as np

from .utils import to_list

from ..batchflow import NamedExpression
from ..batchflow.batchflow.named_expr import _DummyBatch



class Component:
    def __init__(self, component):
        self.component = component

    def __getattr__(self, name):
        return np.array([getattr(val, name) for val in self.component])

    def __setattr__(self, name, value):
        if name == 'component':
            self.__dict__[name] = value
        else:
            if len(self.component) != len(value):
                raise ValueError("Given `value`'s length must be equal to batch size.")
            for item, val in zip(self.component, value):
                setattr(item, name, val)

    def __getitem__(self, key):
        # note, the __getitem__ method must be overridden in the component's class.
        return np.array([val[key] for val in self.component])

    def __setitem__(self, key, value):
        key = to_list(key)
        if len(self.component) != len(value):
                raise ValueError("Given `value`'s length must be equal to batch size.")
        for item, val in zip(self.component, value):
            # note, the __setitem__ method must be overridden in the component's class.
            item[key] = val



class SU(NamedExpression):
    """ !! """
    def get(self, batch=None, pipeline=None, model=None):
        """ !! """
        if self.params:
            batch, pipeline, model = self.params
        name = super()._get_name(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            # TODO: rewrite messsage
            raise ValueError("Batch expressions are not allowed in static model")
        return Component(getattr(batch, name))

    def assign(self, value, batch=None, pipeline=None, model=None):
        if self.params:
            batch, pipeline, model = self.params
        name = super()._get_name(batch=batch, pipeline=pipeline, model=model)
        if name is not None:
            setattr(batch, name, value)
