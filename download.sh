#!/bin/bash
mkdir -p data
mkdir -p data/svhn
mkdir -p data/mnist

if [ ! -f "data/mnist/train-images-idx3-ubyte.gz" ]
then
    wget -O data/mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
else 
    echo "data/mnist/train-images_idx3-ubyte.gz already exist!"
fi

if [ ! -f "data/mnist/train-labels-idx1-ubyte.gz" ]
then
    wget -O data/mnist/train-labels-idx1-ubyte.gz  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
else 
    echo "data/mnist/train-labels-idx1-ubyte.gz already exist!"
fi


if [ ! -f "data/mnist/t10k-labels-idx1-ubyte.gz" ]
then
    wget -O data/mnist/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
else 
    echo "data/mnist/t10k-labels-idx1-ubyte.gz already exist!"
fi

if [ ! -f "data/mnist/t10k-images-idx3-ubyte.gz" ]
then
    wget -O data/mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
else 
    echo "data/mnist/t10k-images-idx3-ubyte.gz already exist!"
fi

if [ ! -f "data/svhn/train_32x32.mat" ]
then
    wget -O data/svhn/train_32x32.mat http://ufldl.stanford.edu/housenumbers/train_32x32.mat
else 
    echo "data/svhn/train_32x32.mat already exist!"
fi


if [ ! -f "data/svhn/test_32x32.mat" ]
then
    wget -O data/svhn/test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
else 
    echo "data/svhn/test_32x32.mat already exist!"
fi

if [ ! -f "data/svhn/extra_32x32.mat" ]
then
    wget -O data/svhn/extra_32x32.mat http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
else 
    echo "data/svhn/extra_32x32.mat already exist!"
fi

