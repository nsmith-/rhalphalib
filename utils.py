import os
import errno
import numpy as np


def make_dirs(dirname):
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_fixed_mins_maxs(mins, maxs):
    deltas = (maxs - mins) / 12.
    mins = mins + deltas / 4.
    maxs = maxs - deltas / 4.

    return [mins, maxs]


def pad2d(arr):
    # Pad 2d array on all sides
    ret = np.zeros(tuple(np.array(arr.shape)+2))
    ret[1:-1, 1:-1] = arr
    ret[0, 1:-1] = arr[0, :]
    ret[-1, 1:-1] = arr[-1, :]
    ret[1:-1, 0] = arr[:, 0]
    ret[1:-1, -1] = arr[:, -1]
    ret[0, 0] = arr[0, 0]
    ret[-1, -1] = arr[-1, -1]
    ret[0, -1] = arr[0, -1]
    ret[-1, 0] = arr[-1, 0]
    return ret