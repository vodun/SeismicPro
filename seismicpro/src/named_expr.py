import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '..')

from seismicpro.batchflow import NamedExpression
from seismicpro.batchflow.batchflow.named_expr import _DummyBatch


class ObjectAttr:
    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        return np.array([getattr(val, name) for val in self.value])

    def __setattr__(self, name, value):
        if name is 'value':
            self.__dict__[name] = value
        else:
            for item, val in zip(self.value, value):
                setattr(item, name, val)

    def __getitem__(self, key):
        return np.array([val.headers[key].values for val in self.value])

    def __setitem__(self, key, value):
        key = np.array(key).ravel().tolist()
        for item, val in zip(self.value, value):
            val = pd.DataFrame(val, columns=key, index=item.headers.index)
            item.headers[key] = val



class SU(NamedExpression):
    """ !! """
    def __init__(self, name=None, mode='w'):
        super().__init__(name, mode)

    def get(self, batch=None, pipeline=None, model=None):
        """ !! """
        if self.params:
            batch, pipeline, model = self.params
        name = super()._get_name(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            raise ValueError("Batch expressions are not allowed in static models: B('%s')" % name)
        return ObjectAttr(getattr(batch, name))

    def assign(self, value, **kwargs):#, batch=None, pipeline=None, model=None):
        """ !! """
        #probably not needed.


