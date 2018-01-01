from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import matplotlib.pyplot as plt
import os, gzip
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim


def rgb2gray(rgb) :
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])

def one_hot_encoding (size, dim, y) :
    one_hot = np.zeros((size,dim))
    one_hot[np.arange(size), y] = 1.0
    return one_hot

def load_svhn(dataset_name,use='train',gray=False): # opt True for using extra
    def __store_data(data, num_of_examples, gray):
        d = []
        for i in range(num_of_examples):
            if gray:
                d.append(__rgb2gray(data[:, :, :, i]))
            else:
                d.append(data[:, :, :, i])
        return np.asarray(d)
    
    #load train set
    data_dir=os.path.join("./data",dataset_name)
    if use=='train'
        train = sio.loadmat(data_dir + "/train_32x32.mat")
        train_size = train['X'].shape[3]
        train_labels = one_hot_encoding(train_size,11,train['y']) # train['y'] has 10
        train_data = __store_data(train['X'].astype("float32"),train_size,gray)
        return train_data/255. , train_labels
    elif use=='test' :
        test = sio.loadmat(data_dir + "/test_32x32.mat")
        test_size = test['X'].shape[3]
        test_labels = one_hot_encoding(train_size,11,test['y'])
        test_data = __store_data(test['X'].astype("float32"),test_size,gray)
        return test_data/255. , test_labels
    else
        extra= sio.loadmat(data_dir + "/extra_32x32.mat")
        extra_size = train['X'].shape[3]
        extra_labels = one_hot_encoding(extra_size,11,extra['y']) # train['y'] has 10
        extra_data = __store_data(extra['X'].astype("float32"),extra_size,gray)
        return extra_data/255. , extra_labels

def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


