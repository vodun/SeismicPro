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


def optimize(matrix, target, weights, batch_size, n_epochs, device='cuda:0', dtype=torch.float32, plot_loss=False,
             optimizer_kwargs=None, sch_kwargs=None, name=None):
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
    for j in tqdm(range(n_epochs)):
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
    np.savez(f'loss_{name}.npz')
    return weights.ravel().detach().cpu().numpy()
