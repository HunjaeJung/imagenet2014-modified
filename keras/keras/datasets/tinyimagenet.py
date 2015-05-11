# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
from skimage import data
import numpy as np

def read_files(path):
    """
    yield data_type(train? val? test?), numpy.ndarray('uint8')
    """
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if not '.txt' in file_name:
                label = file_name.split("_")[0]
                yield label, data.imread(file_path)

def load_data(path):
    """
    load data from tiny-imagenet
    note that in validation set, label information is in val_annotations.txt
    """
    train_size = 100000
    val_size = 10000
    test_size = 10000

    X_train = np.zeros((train_size, 3, 64, 64), dtype="uint8")
    y_train = np.zeros((train_size,), dtype="uint8")
    X_val = np.zeros((val_size, 3, 64, 64), dtype="uint8")
    y_val = np.zeros((val_size,), dtype="uint8")

    path_train = os.path.join(path, 'train')
    path_val = os.path.join(path, 'val')

    for idx, (label, img) in enumerate(read_files(path_train)):
        from IPython import embed; embed()
        X_train[idx, :, :, :] = img
        y_train[idx] = label

    #TODO load validation set from tiny-imagenet

    return (X_train, y_train), (X_val, y_val)


def test_load_data():
    load_data('/shared/tiny-imagenet-200/')


if __name__ == '__main__':
    test_load_data()
