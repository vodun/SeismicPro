from random import randrange
from queue import Queue
from threading import Thread

import numpy as np
from numba import njit


class TensorDataLoader:
    def __init__(self, *tensors, batch_size=1, n_epochs=1, shuffle=False, drop_last=False, device=None):
        if any(len(tensor) != len(tensors[0]) for tensor in tensors):
            raise ValueError("All tensors must have the same length")
        self.tensors = tensors
        self.device = device
        self.n_items = len(tensors[0])
        self.batch_size = min(batch_size, self.n_items)
        self.n_batches_per_epoch, remainder = divmod(self.n_items, self.batch_size)
        if remainder and not drop_last:
            self.n_batches_per_epoch += 1
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        return self.n_batches_per_epoch * self.n_epochs

    def __iter__(self):
        if self.shuffle:
            return ShuffledTensorDataIterator(self)
        return SequentialTensorDataIterator(self)


class BaseTensorDataIterator:
    def __init__(self, loader):
        self.loader = loader
        self.batch_queue = Queue(maxsize=1)
        self.batch_generator_thread = Thread(target=self.gen_batch).start()

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.batch_queue.get(block=True)
        self.batch_queue.task_done()
        if batch is None:
            raise StopIteration
        return batch

    def gen_batch_indices(self):
        raise NotImplementedError

    def gen_batch(self):
        for batch_ix in self.gen_batch_indices():
            batch = tuple(tensor[batch_ix] for tensor in self.loader.tensors)
            if self.loader.device is not None:
                batch = tuple(tensor.to(self.loader.device, non_blocking=True) for tensor in batch)
            self.batch_queue.put(batch, block=True)
        self.batch_queue.put(None, block=True)


class SequentialTensorDataIterator(BaseTensorDataIterator):
    def gen_batch_indices(self):
        for _ in range(self.loader.n_epochs):
            for batch_ix in range(self.loader.n_batches_per_epoch):
                yield slice(batch_ix * self.loader.batch_size, (batch_ix + 1) * self.loader.batch_size)


class ShuffledTensorDataIterator(BaseTensorDataIterator):
    def __init__(self, loader):
        # Precompile numba generator before running it in a thread
        _ = list(self.shuffle_gen(np.arange(3, dtype=np.int32), batch_size=2, drop_last=False))
        self.order = np.arange(loader.n_items, dtype=np.int32)
        super().__init__(loader)

    def gen_batch_indices(self):
        for _ in range(self.loader.n_epochs):
            yield from self.shuffle_gen(self.order, self.loader.batch_size, self.loader.drop_last)

    @staticmethod
    @njit(nogil=True)
    def shuffle_gen(order, batch_size=1, drop_last=False):
        n_items = len(order)
        n_batches, mod = divmod(n_items, batch_size)
        for batch_ix in range(n_batches):
            for item_ix in range(batch_size):
                i = batch_ix * batch_size + item_ix
                j = randrange(i, n_items)
                order[i], order[j] = order[j], order[i]
            yield order[batch_ix * batch_size : (batch_ix + 1) * batch_size]

        for item_ix in range(mod - 1):
            i = n_batches * batch_size + item_ix
            j = randrange(i, n_items)
            order[i], order[j] = order[j], order[i]

        if mod and not drop_last:
            yield order[-mod:]
