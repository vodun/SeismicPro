""" Helper functions for KK and notch filtering """

from time import time

import os
import contextlib
import tempfile

import psutil
import h5py

import numpy as np
from scipy import fft
import torch

from seismiqb import SeismicGeometry


def get_chunk_size(geometries, axis, itemsizes=None, device=None):
    if device:
        mem_avail = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
    else:
        mem_avail = psutil.virtual_memory().available

    if not isinstance(geometries, (list, tuple)):
        geometries = (geometries,)

    if not isinstance(itemsizes, (list, tuple)):
        itemsizes = [itemsizes] * len(geometries)
    elif len(itemsizes) != len(geometries):
        raise ValueError("sizes of geometries and itemsizes lists mismatch!")

    mem_needed_for_slide = 0
    for geom, itemsize in zip(geometries, itemsizes):
        l = list(geom.cube_shape)
        del l[axis]

        if itemsize is None:
            t = geom[0, 0, :1]
            itemsize = t.itemsize

        mem_needed_for_slide += itemsize * l[0] * l[1]

    return mem_avail // mem_needed_for_slide


def get_device(no_gpu):
    device = None
    if not no_gpu and torch.cuda.is_available():
        device = torch.device('cuda')

    print("Using {}".format(device or "cpu"))
    return device

def fk_amp(arr, device):
    return fk_amp_numpy(arr) if device is None else fk_amp_torch_cuda(arr, device)

def fk_amp_numpy(arr):
    """ Calculates amplitude spectrum of a 2D array in f-k domain using numpy"""
    return np.fft.fft2(arr)


def fk_amp_torch_cuda(arr, device):
    """ Calculates amplitude spectrum of a 2D array in f-k domain using cuda with torch"""
    t = torch.tensor(arr, device=device)
    t = torch.stack((t, torch.zeros_like(t)), axis=-1)
    fftt = torch.fft(t, 2)
    res = fftt.cpu().numpy()
    return res


def roll_n(arr, shift, axis):
    """
    Roll array elements along a given axis. Elements that roll beyond the last position are re-introduced at the first.
    adapted from https://github.com/locuslab/pytorch_fft
    """
    f_idx = tuple(slice(None) if i != axis else slice(0, shift) for i in range(arr.dim()))
    b_idx = tuple(slice(None) if i != axis else slice(shift, None) for i in range(arr.dim()))
    front = arr[f_idx]
    back = arr[b_idx]
    return torch.cat([back, front], axis)

def fftshift(arr, dims):
    """ fftshift analogue for torch """
    dims = (dims,) if isinstance(dims, int) else dims
    for dim in dims:
        dim_len = arr.size(dim)
        arr = roll_n(arr, axis=dim, shift=(dim_len//2 + dim_len%2))
    return arr

def apply_sectors(arr, x_off, dx, y_off, dy):
    sx, sy = arr.shape[:2]
    filtered = sectors_i(arr, x_off, dx, sx, sy)
    filtered = sectorsT_i(filtered, y_off, dy, sy, sx)
    return filtered

def sectors_i(arr, dx, dy, sx, sy):
    dd = - dy - dx

    D = 2 * dy - dx
    y = dy

    for i in range(0, dx):
        arr[sx // 2 - i, 0:y+1] = 0
        arr[sx // 2 + i, 0:y+1] = 0

        arr[sx // 2 - i, (sy - y):sy] = 0
        arr[sx // 2 + i, (sy - y):sy] = 0

        if D > 0:
            y = y - 1
            D = D + 2 * dd
        else:
            D = D + 2 * dy

    return arr


def sectorsT_i(arr, dy, dx, sy, sx):
    dd = dx - dy

    D = 2 * dx - dy
    x = dx
    for i in range(0, dy):
        arr[0:x+1, sy // 2 - i] = 0
        arr[0:x+1, sy // 2 + i] = 0

        arr[(sx -x):sx, sy // 2 - i] = 0
        arr[(sx -x):sx, sy // 2 + i] = 0

        if D > 0:
            x = x - 1
            D = D + 2 * dd
        else:
            D = D + 2 * dx

    return arr



def process_kk_notch(inp_path, save_to, kk_params, notch_params, device):
    t0 = time()

    if inp_path.endswith('.sgy'):
        geometry = SeismicGeometry(inp_path)
        geometry.make_hdf5()
        inp_path = os.path.splitext(inp_path)[0]+'.hdf5'

    geometry = SeismicGeometry(inp_path)

    print("Loaded in", time() - t0, flush=True)

    print(geometry, flush=True)

    max_depth = geometry.cube_shape[-1]

    print("default workers:", fft.get_workers())

    if not os.path.exists(os.path.dirname(save_to) or './'):
        os.mkdir(os.path.dirname(save_to))

    with tempfile.TemporaryDirectory() as tmpdirname:
        path_hdf5 = os.path.join(tmpdirname, "tmp.hdf5")

        ctx_mgr = contextlib.suppress() if device else fft.set_workers(25)

        with ctx_mgr:
            print("new workers:", fft.get_workers(), flush=True)
            with h5py.File(path_hdf5, "a") as file_hdf5:
                cube_hdf5 = file_hdf5.create_dataset('cube', geometry.cube_shape)

                i = 0
                while i < max_depth:

                    if device:
                        n_slices =  get_chunk_size(geometry, axis=2, itemsizes=None, device=device) // 8
                    else:
                        n_slices =  get_chunk_size(geometry, axis=2, itemsizes=np.dtype(np.complex128).itemsize) // 2

                    print("chunk size:", n_slices, flush=True)

                    t1 = time()
                    slices = geometry[:, :, i:min(max_depth, i + n_slices)]
                    print("\tread in:", time()-t1, flush=True)

                    t1 = time()
                    res = do_filter(slices, kk_params, notch_params, device=device)
                    print("\tfiltered in:", time()-t1, flush=True)

                    t1 = time()
                    cube_hdf5[:, :, i:i+res.shape[-1]] = res
                    print("\twritten in:", time()-t1, flush=True)

                    i = i + n_slices

        print("Filtered in", time() - t0, flush=True)

        geometry.make_sgy(path_hdf5=path_hdf5, path_segy=save_to,
                          path_spec=os.path.splitext(inp_path)[0]+'.sgy',
                          zip_result=False)

    print("Done in", time() - t0)

def do_filter(slices, kk_params, notch_params, device=None):
    if device:
        t = torch.tensor(slices, device=device).permute(2, 0, 1)
        t = torch.stack((t, torch.zeros_like(t)), axis=-1)

        fftt = torch.fft(t, 2).permute(1, 2, 3, 0)
        del t

        if kk_params:
            fftt = apply_sectors(fftt, *kk_params)
        if notch_params:
            fftt = apply_notch(fftt, *notch_params)

        fftt = fftt.permute(3, 0, 1, 2)
        res = torch.ifft(fftt, 2)
        del fftt

        res = res[:, :, :, 0].permute(1, 2, 0)
        return res.cpu().numpy()
    else:
        fft_slices = fft.fft2(slices, axes=(0, 1))

        if notch_params:
            fft_slices = apply_notch(fft_slices, *notch_params)
        if kk_params:
            fft_slices = apply_sectors(fft_slices, *kk_params)

        res = fft.ifft2(fft_slices, axes=(0, 1)).real
        return res

def notch_filter(slices, x_off, dx, y_off, dy):

    fft_slices = fft.fft2(slices, axes=(0, 1))
    filtered = apply_notch(fft_slices, x_off, dx, y_off, dy)
    res = fft.ifft2(filtered, axes=(0, 1)).real
    return res

def apply_notch(arr, x_off, dx, y_off, dy):

    sx, sy = arr.shape[:2]

    brdr, fill = ellipse_filters(dx, dy)
    flt_b = np.full((sx, sy), False)
    flt_f = np.full((sx, sy), False)

    for i in range(sx//x_off//2):
        for j in range(sy//y_off//2):
            if i == 0 and j == 0:
                continue

            for x0 in (i*x_off, sx-i*x_off-1):
                for y0 in (j*y_off, sy-j*y_off-1):
                    shift_filter(brdr, dx, dy, flt_b, x0, y0, sx, sy)
                    shift_filter(fill, dx, dy, flt_f, x0, y0, sx, sy)

                    arr[flt_f] = np.mean(arr[flt_b], axis=0)

                    flt_b[:] = False
                    flt_f[:] = False

    return arr


def shift_filter(flt, a, b, res, x0, y0, sx, sy):
    f0x = max(0, a - x0)
    f0y = max(0, b - y0)

    r0x = max(0, x0 - a)
    r0y = max(0, y0 - b)


    f1x = min(a + sx - x0, 2*a + 1)
    f1y = min(b + sy - y0, 2*b + 1)

    r1x = min(x0 + a + 1, sx)
    r1y = min(y0 + b + 1, sy)

    res[r0x:r1x, r0y:r1y] = flt[f0x:f1x, f0y:f1y]


def set_filters(brdr, fill, x, y0, y1, sx, sy):
    if 0 <= x < sx:
        brdr[x, max(0, y0)] = True
        brdr[x, min(y1, sy)-1] = True
        fill[x, max(0, y0):min(y1, sy)] = True


def ellipse_filters(a, b):

    sx, sy = a*2+1, b*2+1
    x0 = a
    y0 = b
    brdr = np.full((sx, sy), False)
    fill = np.full((sx, sy), False)

    dx = b*b - 2 * b *b * a
    dy = a * a
    Exy = 0
    x = a
    y = 0

    while x > 0:
        set_filters(brdr, fill, x0 + x, y0 - y, y + y0 + 1, sx, sy)
        set_filters(brdr, fill, x0 - x, y0 - y, y + y0 + 1, sx, sy)

        step_x = False
        step_y = False

        if 2*Exy + 2*dx + dy <= 0:
            step_y = True
        if 2*Exy + 2*dy + dx >= 0:
            step_x = True

        if step_x:
            x -= 1
            Exy += dx
            dx += 2*b*b

        if step_y:
            y += 1
            Exy += dy
            dy += 2*a*a

    set_filters(brdr, fill, x0, y0 - y, y + y0 + 1, sx, sy)

    return brdr, fill
