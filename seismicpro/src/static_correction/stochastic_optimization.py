import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from IPython.display import clear_output


class SparseDataset(Dataset):
    def __init__(self, matrix, y, dtype, device):
        if sparse.isspmatrix_coo(matrix):
            raise TypeError("matrix cannot have COO format")
        self.matrix = matrix
        self.y = y
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, ixs):
        traces = self.matrix[ixs]
        tensor = torch.sparse_coo_tensor(traces.nonzero(), traces.data,
                                         dtype=self.dtype, device=self.device, size=traces.shape)
        y = self.y[ixs]
        y = torch.tensor(y, dtype=self.dtype, device=self.device)
        return tensor, y



def optimize(matrix, target, weights, norms, batch_size, n_epochs, device='cuda:0', dtype=torch.float32, plot_loss=False,
             optimizer_kwargs=None, sch_kwargs=None, cp_folder=None, cp_frequency=None):
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    sch_kwargs = {} if sch_kwargs is None else sch_kwargs

    sparse_dataset = SparseDataset(matrix, target, dtype, device)

    weights = torch.tensor(weights, dtype=dtype, device=device, requires_grad=True)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW([weights], **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sch_kwargs)

    indices = np.arange(0, len(sparse_dataset))
    pos_ix = np.arange(0, len(sparse_dataset)+batch_size, batch_size).astype(np.int32)
    pos_ix = tuple(zip(pos_ix[: -1], pos_ix[1:]))

    loss_list = []
    for n_epoch in tqdm(range(n_epochs)):
        np.random.shuffle(indices)
        for i, idx in enumerate(pos_ix):
            sub_traces, y = sparse_dataset[indices[slice(*idx)]]
            y_pred = sub_traces.matmul(weights)
            optimizer.zero_grad()
            loss = loss_fn(y_pred.ravel(), y)
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()

        if plot_loss:
            clear_output()
            print(f"Loss: {loss_list[-1]}")
            plt.plot(loss_list)
            plt.show()
        scheduler.step()

        if cp_frequency is not None and (n_epoch % cp_frequency==0 or n_epoch == n_epochs-1):
            n_epoch = n_epoch if n_epoch != n_epochs-1 else 'LAST'
            dump_weights = weights.ravel().detach().cpu().numpy() / norms
            add_checkpoint(folder=cp_folder, weights=dump_weights, loss=loss_list, n_epoch=n_epoch)

    return weights.ravel().detach().cpu().numpy() / norms


def add_checkpoint(folder, weights, loss, n_epoch):
    if not os.path.exists(folder):
        os.makedirs(folder)
    weights_path = os.path.join(folder, f"{n_epoch}_epoch_weights.npz")
    loss_path = os.path.join(folder, f"{n_epoch}_epoch_loss.npz")
    np.savez(weights_path, weights)
    np.savez(loss_path, loss)
