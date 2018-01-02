import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def one_hot_encoding(size, dim, y):
    one_hot = np.zeros((size, dim))
    one_hot[np.arange(size), y] = 1.0
    return one_hot
