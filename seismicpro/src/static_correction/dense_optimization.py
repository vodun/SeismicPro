import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from IPython.display import clear_output


class DenseDataset(Dataset):
    def __init__(self, tr_events_ixs, tr_grid, source_coefs, upholes, rec_coefs, weights, targets, dtype, device):
        self.tr_events_ixs = tr_events_ixs
        self.tr_grid = torch.tensor(tr_grid, dtype=dtype, device=device)
        self.source_coefs = torch.tensor(source_coefs, dtype=dtype, device=device)
        self.upholes = torch.tensor(upholes, dtype=dtype, device=device)
        self.rec_coefs = torch.tensor(rec_coefs, dtype=dtype, device=device)
        self.weights = weights

        self.targets = torch.tensor(targets, dtype=dtype, device=device)

        self.dtype = dtype
        self.device = device

    def __len__(self):
        return self.tr_events_ixs.shape[0]

    def __getitem__(self, ixs):
        scoef_ixs, rcoef_ixs = self.tr_events_ixs[ixs].T
        targets = self.targets[ixs]
        grid = self.tr_grid[ixs]
        upholes = self.upholes[ixs]

        source_coefs = self.source_coefs[scoef_ixs]
        rec_coefs = self.rec_coefs[rcoef_ixs]

        weights = F.grid_sample(self.weights[None, None], torch.vstack((grid[:, :2], grid[:, 2:]))[None, None],
                                align_corners=True, mode="bicubic").ravel()
        return source_coefs, weights[:len(ixs)] - upholes, rec_coefs, weights[len(ixs): ], targets



def dense_optimize(tr_events_ixs, tr_coords, source_coefs, upholes, rec_coefs, weights, targets, batch_size, n_epochs,
                   device='cuda:0', dtype=torch.float32, plot_loss=False, optimizer_kwargs=None, sch_kwargs=None,
                   cp_folder=None, cp_frequency=None):
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    sch_kwargs = {"step_size": np.inf} if sch_kwargs is None else sch_kwargs

    dataset = DenseDataset(tr_events_ixs, tr_coords, source_coefs, upholes, rec_coefs, weights,  targets,
                           dtype=dtype, device=device)

    weights = torch.tensor(weights, dtype=dtype, device=device, requires_grad=True)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([weights], **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sch_kwargs)

    loss_list = []
    indices = np.arange(0, targets.shape[0])

    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        for i in range(0, targets.shape[0], batch_size):
            shot_c, shot_w, rec_c, rec_w, y = dataset[indices[i: i+batch_size]]
            y_pred = shot_c*shot_w + rec_c*rec_w
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y)
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
            dump_weights = weights.ravel().detach().cpu().numpy()
            add_checkpoint(folder=cp_folder, weights=dump_weights, loss=loss_list, n_epoch=n_epoch)

    return weights


def add_checkpoint(folder, weights, loss, n_epoch):
    if not os.path.exists(folder):
        os.makedirs(folder)
    weights_path = os.path.join(folder, f"{n_epoch}_epoch_weights.npz")
    loss_path = os.path.join(folder, f"{n_epoch}_epoch_loss.npz")
    np.savez(weights_path, weights)
    np.savez(loss_path, loss)
