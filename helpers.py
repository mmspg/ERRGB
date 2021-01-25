"""
This module contains all helper functions.
"""
import numpy as np


def transform(x, means, stdvs):
    """
    Transform the input data of the aligned reID network to have means and stdvs
    on the different RGB channels as used during training.

    :param x:       input image
    :param means:   target means, [mean_R, mean_G, mean_B]
    :param stdvs:   target stdvs, [stdv_R, stdv_G, stdv_B]

    :return:        transformed input with mean = target mean, stdv = target stdv
    """
    # normalize along the channels
    x_squeezed = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    old_mean = np.mean(x_squeezed, 1)
    old_stdv = np.std(x_squeezed, 1)
    x[0, :, :] = (x[0, :, :] - old_mean[0]) * (stdvs[0] / old_stdv[0]) + means[0]
    x[1, :, :] = (x[1, :, :] - old_mean[1]) * (stdvs[1] / old_stdv[1]) + means[1]
    x[2, :, :] = (x[2, :, :] - old_mean[2]) * (stdvs[2] / old_stdv[2]) + means[2]

    x_squeezed = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    new_mean = np.mean(x_squeezed, 1)
    new_stdv = np.std(x_squeezed, 1)
    return x


def normalize(vec):
    """
    Normalizes the input vector.

    :param vec: input vec

    :return:    normalized vec
    """
    min_vec = np.min(vec)
    max_vec = np.max(vec)
    return (vec - min_vec) / (max_vec - min_vec)
