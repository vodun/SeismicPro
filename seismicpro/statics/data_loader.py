import torch


class TensorDataLoader:
    def __init__(self, *tensors, batch_size=1, shuffle=False, drop_last=False, device=None):
        if any(len(tensor) != len(tensors[0]) for tensor in tensors):
            raise ValueError("All tensors must have the same length")
        self.tensors = tensors
        self.device = device
        self.n_items = len(tensors[0])
        self.batch_size = min(batch_size, self.n_items)
        self.shuffle = shuffle
        self.n_batches, remainder = divmod(self.n_items, self.batch_size)
        if remainder and not drop_last:
            self.n_batches += 1
        self.current_batch = 0

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n_items)
            self.tensors = tuple(tensor[indices] for tensor in self.tensors)
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.n_batches:
            raise StopIteration
        batch = tuple(tensor[self.current_batch * self.batch_size : (self.current_batch + 1) * self.batch_size]
                      for tensor in self.tensors)
        if self.device is not None:
            batch = tuple(tensor.to(self.device) for tensor in batch)
        self.current_batch += 1
        return batch
