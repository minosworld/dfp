import ctypes
import glob
import operator
import os

from multiprocessing import sharedctypes
import matplotlib.pyplot as plt
import numpy as np


def make_objective_indices_and_coeffs(temporal_coeffs, meas_coeffs):
    objective_coeffs = (np.reshape(temporal_coeffs, (1,-1)) * np.reshape(meas_coeffs, (-1,1))).flatten()
    objective_indices = np.where(np.abs(objective_coeffs) > 1e-8)[0]
    return objective_indices, objective_coeffs[objective_indices]


def make_array(shape=(1,), dtype=np.float32, shared=False, fill_val=None):  
    np_type_to_ctype = {np.float32: ctypes.c_float,
                        np.float64: ctypes.c_double,
                        np.bool: ctypes.c_bool,
                        np.uint8: ctypes.c_ubyte,
                        np.uint64: ctypes.c_ulonglong}
    if not shared:
        np_arr = np.empty(shape, dtype=dtype)
    else:
        numel = np.prod(shape)
        arr_ctypes = sharedctypes.RawArray(np_type_to_ctype[dtype], numel)
        np_arr = np.frombuffer(arr_ctypes, dtype=dtype, count=numel)
        np_arr.shape = shape

    if not fill_val is None:
        np_arr[...] = fill_val

    return np_arr


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


def list_checkpoints(checkpoint_dir, range_str=None):
    if os.path.isfile(checkpoint_dir):
        path = os.path.splitext(checkpoint_dir)[0]
        step = int(path.split('-')[1])
        return {step: path}
    ckpt_range = list(map(int, range_str.lower().replace('k','000').split('-'))) if range_str else None
    ckpt_filenames = glob.glob(checkpoint_dir + '/*.index')
    ckpts = {}
    for ckpt_filename in ckpt_filenames:
        base = os.path.splitext(ckpt_filename)[0]
        step = int(base.split('-')[1])
        if isinstance(ckpt_range, list) and (step < ckpt_range[0] or step > ckpt_range[1]):
            continue
        ckpts[step] = base
    return ckpts


class StackedBarPlot:
    def __init__(self, data, nfig=17, labels=[], ylim=[]):
        self.data = data
        self.fig = plt.figure(nfig)
        self.ax = plt.gca()
        self.colors = ['r','g','b','y','c','m','k', [0.5,0,0], [0,0.5,0], [0,0,0.5], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [0.5,0.5,0.5]]
        self.xs = np.arange(data.shape[1])
        self.width = 0.9

        if len(labels):
            assert(len(labels) == data.shape[1])
            self.labels = labels
        else:
            self.labels = range(data.shape[1])

        if len(ylim):
            self.ax.set_ylim(ylim)

        self.plots = []
        curr_bot = np.zeros(data.shape[1])
        for n in range(data.shape[0]):
            self.plots.append(plt.bar(self.xs, data[n], bottom=curr_bot, width=self.width, color=self.colors[n%len(self.colors)]))
            curr_bot += data[n]

        plt.ylabel('Objective value')
        plt.title('Action selection')
        plt.xticks(self.xs + self.width/2., self.labels)

    def show(self):
        self.fig.show()

    def draw(self):
        self.fig.canvas.draw()

    def set_data(self, data, labels=[]):
        assert(data.shape == self.data.shape)
        curr_bot = np.zeros(data.shape[1])
        for n in range(data.shape[0]):
            for nr in range(data.shape[1]):
                self.plots[n][nr].set_height(data[n,nr])
                self.plots[n][nr].set_y(curr_bot[nr])
            curr_bot += data[n]
        if len(labels):
            assert(len(labels) == data.shape[1])
            self.labels = labels
            self.ax.set_xticklabels(self.labels)
