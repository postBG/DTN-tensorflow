import tensorflow as tf
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def one_hot_encoding(size, dim, y):
    labels = tf.fill([size, 1], y)
    return tf.reshape(tf.one_hot(labels, dim), [-1, dim])
