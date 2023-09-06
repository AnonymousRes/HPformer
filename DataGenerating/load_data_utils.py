from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from DataGenerating import common_utils
from tqdm import tqdm


def get_bin_custom(x, nbins):
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    for i in range(nbins):
        a = bins[i][0] * 24.0
        b = bins[i][1] * 24.0
        if a <= x < b:
            ret = [0.0] * nbins
            ret[i] = 1.0
            return ret


def dp_load_data(dataloader, discretizer, normalizer=None):
    timestep = discretizer._timestep
    def get_bin(t):
        eps = 1e-6
        return int(t / timestep - eps)
    N = len(dataloader._data["X"])
    Xs = []
    ts = []
    masks = []
    curmasks = []
    ys = []
    names = []

    for i in tqdm(range(N), desc='Discretizer'):
        X = dataloader._data["X"][i]
        cur_ts = dataloader._data["ts"][i]
        cur_ys = dataloader._data["ys"][i]
        name = dataloader._data["name"][i]

        cur_ys = [int(x) for x in cur_ys]

        T = max(cur_ts)
        nsteps = get_bin(T) + 1
        mask = [1] * nsteps
        curmask = [0] * nsteps
        curmask[-1] = 1
        y = [0] * nsteps

        for pos, z in zip(cur_ts, cur_ys):
            y[get_bin(pos)] = z

        # check label
        if int(y[4])==1:
            if nsteps <= 24:
                y[:4] = [1, 1, 1, 1]
            if nsteps > 24:
                if nsteps == 25:
                    y[1:4] = [1, 1, 1]
                if nsteps == 26:
                    y[2:4] = [1, 1]
                if nsteps == 27:
                    y[3] = 1


        X = discretizer.transform(X, end=T)[0]
        Xs.append(X)
        masks.append(np.array(mask))
        curmasks.append(np.array(curmask))
        ys.append(np.array(y))
        names.append(name)
        ts.append(cur_ts)
        assert np.sum(mask) > 0
        assert len(X) == len(mask) and len(X) == len(y)

    yys = [np.expand_dims(yyi, axis=-1) for yyi in ys] #(B, T, 1)
    if normalizer is None:
        # normalizer
        print('Building normalizer...')
        X_norm = []
        for p in Xs:
            for row in p:
                X_norm.append(row)
        X_norm = np.array(X_norm, dtype=float)
        normalizer_ = StandardScaler()
        normalizer_.fit(X_norm)
        data = [normalizer_.transform(X) for X in Xs]
        print('X_norm_shape:', X_norm.shape)
        print('normalizer_mean:', normalizer_.mean_)
        finaldata = [[data, masks, curmasks, names], yys]
        return finaldata, normalizer_
    else:
        normalizer_= normalizer
        data = [normalizer_.transform(X) for X in Xs]
        finaldata = [[data, masks, curmasks, names], yys]
        return finaldata


def ihmp_load_data(dataloader, discretizer, normalizer=None, small_part=False, return_names=False):
    N = dataloader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(dataloader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    print("Discretizer")
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]

    if normalizer is None:
        # normalizer
        print('Building normalizer...')
        X_norm = []
        for p in data:
            for row in p:
                X_norm.append(row)
        X_norm = np.array(X_norm, dtype=float)
        normalizer_ = StandardScaler()
        normalizer_.fit(X_norm)
        data = [normalizer_.transform(X) for X in data]
        print('X_norm_shape:', X_norm.shape)
        print('normalizer_mean:', normalizer_.mean_)
        finaldata = [np.array(data), labels, names]
        return finaldata, normalizer_
    else:
        normalizer_= normalizer
        data = [normalizer_.transform(X) for X in data]
        finaldata = [np.array(data), labels, names]
        return finaldata


def los_load_data(dataloader, discretizer, normalizer=None):
    timestep = discretizer._timestep
    def get_bin(t):
        eps = 1e-6
        return int(t / timestep - eps)
    N = len(dataloader._data["X"])
    Xs = []
    ts = []
    masks = []
    curmasks = []
    ys = []
    names = []

    for i in tqdm(range(N), desc='Discretizer'):
        X = dataloader._data["X"][i]
        cur_ts = dataloader._data["ts"][i]
        cur_ys = dataloader._data["ys"][i]
        name = dataloader._data["name"][i]
        T = max(cur_ts)
        nsteps = get_bin(T) + 1
        mask = [1] * nsteps
        curmask = [0] * nsteps
        curmask[-1] = 1
        y = [0.0] * nsteps
        for pos, z in zip(cur_ts, cur_ys):
            y[get_bin(pos)] = float(z)

        # check label
        if int(sum(y[:4]))==0:
            y[:4] = [y[4]+4.01, y[4]+3.001, y[4]+2.001, y[4]+1.001]

        y = [get_bin_custom(x, 10) for x in y]
        X = discretizer.transform(X, end=T)[0]
        Xs.append(X)
        masks.append(np.array(mask))
        curmasks.append(np.array(curmask))
        ys.append(np.array(y))
        names.append(name)
        ts.append(cur_ts)
        assert np.sum(mask) > 0
        assert len(X) == len(mask) and len(X) == len(y)

    if normalizer is None:
        # normalizer
        print('Building normalizer...')
        X_norm = []
        for p in Xs:
            for row in p:
                X_norm.append(row)
        X_norm = np.array(X_norm, dtype=float)
        normalizer_ = StandardScaler()
        normalizer_.fit(X_norm)
        data = [normalizer_.transform(X) for X in Xs]
        print('X_norm_shape:', X_norm.shape)
        print('normalizer_mean:', normalizer_.mean_)
        finaldata = [[data, masks, curmasks, names], ys]
        return finaldata, normalizer_
    else:
        normalizer_= normalizer
        data = [normalizer_.transform(X) for X in Xs]
        finaldata = [[data, masks, curmasks, names], ys]
        return finaldata


def phen_load_data(dataloader, discretizer, normalizer=None, small_part=False):
    N = dataloader.get_number_of_examples()
    Xs = []
    masks = []
    curmasks = []
    labels = []
    names = []
    for i in tqdm(range(N), desc='Discretizer'):
        ret = dataloader.read_example(index=i)
        X = ret["X"]
        cur_t = ret["t"]
        cur_y = ret["y"]
        name = ret["name"]
        T = cur_t
        X = discretizer.transform(X, end=T)[0]
        nsteps = len(X)
        mask = [1] * nsteps
        curmask = [0] * nsteps
        curmask[-1] = 1
        Xs.append(X)
        masks.append(np.array(mask))
        curmasks.append(np.array(curmask))
        labels.append(np.array(cur_y))
        names.append(name)
    if normalizer is None:
        # normalizer
        print('Building normalizer...')
        X_norm = []
        for p in Xs:
            for row in p:
                X_norm.append(row)
        X_norm = np.array(X_norm, dtype=float)
        normalizer_ = StandardScaler()
        normalizer_.fit(X_norm)
        data = [normalizer_.transform(X) for X in Xs]
        print('X_norm_shape:', X_norm.shape)
        print('normalizer_mean:', normalizer_.mean_)
        finaldata = [data, masks, curmasks, labels, names]
        return finaldata, normalizer_
    else:
        normalizer_= normalizer
        data = [normalizer_.transform(X) for X in Xs]
        finaldata = [data, masks, curmasks, labels, names]
        return finaldata