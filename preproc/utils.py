import os
import pickle
import scipy.io as sio
import numpy as np

from utils import rgb2gray
from .preprocess import MNIST_PATH, SVHN_PATH


def load_mnist(data_dir=MNIST_PATH, use='train'):
    """
    Load preprocessed mnist and normalize it to [-1, 1]
    
    parameter
    - data_dir: directory path where mnist pickles exist. default MNIST_PATH 
    - use: load 'train' or 'test'
    
    return: normalized images and labels
    """
    print('loading mnist image dataset..')
    data_file = 'train.pkl' if use == 'train' else 'test.pkl'
    data_dir = os.path.join(data_dir, data_file)

    with open(data_dir, 'rb') as f:
        mnist = pickle.load(f)
    images = mnist['X'] / 127.5 - 1
    labels = mnist['y']

    print('finished loading mnist image dataset..!')
    return images, labels


def load_svhn(data_dir=SVHN_PATH, use='train', gray=False):
    """
    Load preprocessed SVHN and normalize it to [-1, 1]
    
    parameter
    - data_dir : directory path where mnist pickles exist. default SVHN_PATH 
    - use : load 'train', 'test', or 'extra' ( 'extra' is not recommanded becuase of it size 1.2GB )
    - tray : if True then save gray images (boolean)
    
    return: normalized images and labels
    """

    def __store_data(data, num_of_examples, gray):
        """
        Append data by following gray option

        parameter
        - data : the structure which contains images
        - num_of_examples : number of images (integer)
        - gray : if True then save gray images (boolean)
        """

        d = []
        for i in range(num_of_examples):
            if gray:
                d.append(rgb2gray(data[:, :, :, i]))
            else:
                d.append(data[:, :, :, i])
        return np.asarray(d)

    print(os.path.join(data_dir, use + "_32x32.mat"))
    data = sio.loadmat(os.path.join(data_dir, use + "_32x32.mat"))
    data_size = data['X'].shape[3]

    labels = np.zeros((data_size, 10))
    labels[np.arange(data_size), np.transpose((data['y'] % 10))] = 1.0
    images = __store_data(data['X'].astype(np.float32), data_size, gray)
    images = images / 127.5 - 1
    print('finished loading svhn ' + use + ' dataset..!')

    return images, labels

