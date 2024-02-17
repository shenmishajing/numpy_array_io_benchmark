import gzip
import os
import shutil
import time
from glob import iglob

import h5py
import numpy as np
from tqdm import trange


def timer_function(
    func, n=1, name=None, init_func=None, post_func=None, *args, **kwargs
):
    durations = []
    name = func.__name__ if name is None else name
    for _ in trange(n, desc=name):
        if init_func is not None:
            init_func(*args, **kwargs)

        start = time.time()
        func(*args, **kwargs)
        durations.append(time.time() - start)

        if post_func is not None:
            post_func(*args, **kwargs)

    print(f"{name} took {sum(durations)/len(durations)} seconds.")


# tools
def init_func(data_path, *args, **kwargs):
    os.makedirs(data_path, exist_ok=True)


def post_func(data_path, *args, **kwargs):
    shutil.rmtree(data_path, ignore_errors=True)


def get_save_size(data_path, *args, **kwargs):
    return sum(
        os.path.getsize(file)
        for file in iglob(os.path.join(data_path, "**"), recursive=True)
    )


# npy
def npy_save(data, data_path, *args, **kwargs):
    for key, value in data.items():
        np.save(os.path.join(data_path, f"{key}.npy"), value)


def npy_read(data_path, *args, **kwargs):
    res = {}
    for file in iglob(os.path.join(data_path, "*.npy")):
        key = os.path.splitext(os.path.basename(file))[0]
        res[key] = np.load(file)
    return res


def npy_save_gz(data, data_path, *args, **kwargs):
    for key, value in data.items():
        with gzip.GzipFile(os.path.join(data_path, f"{key}.npy.gz"), "w") as f:
            np.save(f, value)


def npy_read_gz(data_path, *args, **kwargs):
    res = {}
    for file in iglob(os.path.join(data_path, "*.npy.gz")):
        key = os.path.splitext(os.path.basename(file))[0]
        with gzip.GzipFile(file, "r") as f:
            res[key] = np.load(f)
    return res


# npz
def npz_save(data, data_path, *args, **kwargs):
    np.savez(os.path.join(data_path, "data.npz"), **data)


def npz_save_compressed(data, data_path, *args, **kwargs):
    np.savez_compressed(os.path.join(data_path, "data.npz"), **data)


def npz_read(data_path, *args, **kwargs):
    res = np.load(os.path.join(data_path, "data.npz"))
    for key in res:
        a = res[key]
    return res


# h5py
def h5py_save(data, data_path, *args, **kwargs):
    with h5py.File(os.path.join(data_path, "data.h5"), "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)


def h5py_read(data_path, *args, **kwargs):
    res = {}
    with h5py.File(os.path.join(data_path, "data.h5"), "r") as f:
        for key in f.keys():
            res[key] = f[key][:]
    return res


def main():
    data_path = "data"
    shape = (4096, 4096)
    keys = "abcefghijk"
    n = 10
    data = {k: np.random.random(shape) for k in keys}

    # read func
    for read_func, save_func in [
        (npy_read, npy_save),
        (npy_read_gz, npy_save_gz),
        (npz_read, npz_save),
        (npz_read, npz_save_compressed),
        (h5py_read, h5py_save),
    ]:
        post_func(data_path)
        init_func(data_path)
        save_func(data, data_path)
        print(
            f"{save_func.__name__} save data as {get_save_size(data_path)/1024**2} MB"
        )

        timer_function(
            read_func,
            n,
            data=data,
            data_path=data_path,
        )

        post_func(data_path)

    # save func
    for save_func in [npy_save, npy_save_gz, npz_save, npz_save_compressed, h5py_save]:
        timer_function(
            save_func,
            n,
            init_func=init_func,
            post_func=post_func,
            data=data,
            data_path=data_path,
        )


if __name__ == "__main__":
    main()
