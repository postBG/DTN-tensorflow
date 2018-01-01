#!/bin/bash
mkdir -p data
mkdir -p data/svhn

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

