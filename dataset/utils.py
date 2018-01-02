import os
import pickle

from dataset.preprocess import MNIST_PATH


def load_mnist(image_dir=MNIST_PATH, use='train'):
    """
    Load preprocessed mnist and normalize it to [-1, 1]

    :param image_dir: directory path where mnist pickles exist. default MNIST_PATH 
    :param use: load 'train' or 'test'
    :return: normalized images and labels
    """
    print('loading mnist image dataset..')
    image_file = 'train.pkl' if use == 'train' else 'test.pkl'
    image_dir = os.path.join(image_dir, image_file)

    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f)
    images = mnist['X'] / 127.5 - 1
    labels = mnist['y']

    print('finished loading mnist image dataset..!')
    return images, labels


def load_svhn(dataset_name, use='train', gray=False):  # opt True for using extra
    # def __store_data(data, num_of_examples, gray):
    #     d = []
    #     for i in range(num_of_examples):
    #         if gray:
    #             d.append(__rgb2gray(data[:, :, :, i]))
    #         else:
    #             d.append(data[:, :, :, i])
    #     return np.asarray(d)
    #
    # # load train set
    # data_dir = os.path.join("./data", dataset_name)
    # if use == 'train'
    #     train = sio.loadmat(data_dir + "/train_32x32.mat")
    #     train_size = train['X'].shape[3]
    #     train_labels = one_hot_encoding(train_size, 11, train['y'])  # train['y'] has 10
    #     train_data = __store_data(train['X'].astype("float32"), train_size, gray)
    #     return train_data / 255., train_labels
    # elif use == 'test':
    #     test = sio.loadmat(data_dir + "/test_32x32.mat")
    #     test_size = test['X'].shape[3]
    #     test_labels = one_hot_encoding(train_size, 11, test['y'])
    #     test_data = __store_data(test['X'].astype("float32"), test_size, gray)
    #     return test_data / 255., test_labels
    # else
    #     extra = sio.loadmat(data_dir + "/extra_32x32.mat")
    #     extra_size = train['X'].shape[3]
    #     extra_labels = one_hot_encoding(extra_size, 11, extra['y'])  # train['y'] has 10
    #     extra_data = __store_data(extra['X'].astype("float32"), extra_size, gray)
    #     return extra_data / 255., extra_labels
    pass
