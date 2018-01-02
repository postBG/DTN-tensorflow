import os
import pickle
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

MNIST_PATH = os.path.join(os.path.dirname(__file__), '../data/mnist')
SVHN_PATH = os.path.join(os.path.dirname(__file__), '../data/svhn')


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' % path)


def resize_images(images_as_array, size=(32, 32)):
    """
    Resize images to size and expand dimension
    
    :param images_as_array: numpy array of pixels (eg) shapes [# of images, 28, 28] 
    :param size: tuple or list of size (eg) (32, 32) or [32, 32]
    :return: images as numpy array shaped [# of images, 32, 32, 3]
    """
    images_as_array = (images_as_array * 255).astype(np.uint8)
    images = [Image.fromarray(image) for image in images_as_array]

    resized_images = [image.resize(size=size, resample=Image.ANTIALIAS) for image in images]
    resized_images_as_array = [np.asarray(image) for image in resized_images]
    resized_images_as_array = np.asarray(resized_images_as_array)

    return np.expand_dims(resized_images_as_array, 3)


def main():
    mnist = input_data.read_data_sets(train_dir=MNIST_PATH)

    train = {'X': resize_images(mnist.train.images.reshape(-1, 28, 28)),
             'y': mnist.train.labels}

    test = {'X': resize_images(mnist.test.images.reshape(-1, 28, 28)),
            'y': mnist.test.labels}

    save_pickle(train, os.path.join(MNIST_PATH, 'train.pkl'))
    save_pickle(test, os.path.join(MNIST_PATH, 'test.pkl'))


if __name__ == "__main__":
    main()
