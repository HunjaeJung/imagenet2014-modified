# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
from skimage import data
from sklearn.preprocessing import LabelEncoder
import numpy as np

def read_files(dir, dataset):
    """
    yield data_type(train? val? test?), numpy.ndarray('uint8')
    """
    dir_path = os.path.join(dir,dataset)
    if dataset=='train':
        for(root, dirs, files) in os.walk(dir_path):
            for file in files:
                if not '.txt' in file:
                    label = file.split("_")[0]
                    img_filepath = os.path.join(root,file)
                    yield label, data.imread(img_filepath)
    elif dataset=='val':
        for(root, dirs, files) in os.walk(dir_path):
            for file in files:
                if '.txt' in file:
                    # this is val_annotaions.txt
                    f = open(os.path.join(root,file), 'r')
                    while 1:
                        line = f.readline()
                        if not line: break
                        line_seg = line.split()
                        img_filepath = os.path.join(root,'images',line_seg[0])
                        label = line_seg[1]
                        yield label, data.imread(img_filepath)
                    f.close()

def load_data(path):
    """
    load data from tiny-imagenet
    note that in validation set, label information is in val_annotations.txt
    """
    train_size = 100000
    val_size = 10000
    test_size = 10000

    # for training data set
    X_train = np.zeros((train_size, 3, 64, 64), dtype="uint8")
    # y_train = np.zeros((train_size,), dtype="str")
    y_train = np.chararray((train_size,), itemsize=10)

    # for validation data set
    X_val = np.zeros((val_size, 3, 64, 64), dtype="uint8")
    # y_val = np.zeros((val_size,), dtype="str")
    y_val = np.chararray((val_size,), itemsize=10)

    #path_train = os.path.join(path, 'train')
    #path_val = os.path.join(path, 'val')

    print "load training data..."
    for idx, (label, img) in enumerate(read_files(path,'train')):
        # reshape (64, 64, 3) -> (3, 64, 64)
        # gray color image is combined ... e.g. n04366367_182.JPEG
        # Grey-scale means that all values have the same intensity. Set all channels
        # (in RGB) equal to the the grey value and you will have the an RGB black and
        # white image.
        if img.ndim == 2:
            img = np.array([img[:, :], img[:, :], img[:, :]])
        elif img.ndim == 3:
            img = np.array([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
        X_train[idx, :, :, :] = img
        y_train[idx] = label

    # change text label(n04366367, ...) to (0, 1, 2, ...)
    print "encoding labels for training data..."
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    print "load validation data..."
    for idx, (label, img) in enumerate(read_files(path,'val')):
        # reshape (64, 64, 3) -> (3, 64, 64)
        # gray color image is combined ... e.g. n04366367_182.JPEG
        # Grey-scale means that all values have the same intensity. Set all channels
        # (in RGB) equal to the the grey value and you will have the an RGB black and
        # white image.
        if img.ndim == 2:
            img = np.array([img[:, :], img[:, :], img[:, :]])
        elif img.ndim == 3:
            img = np.array([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
        X_val[idx, :, :, :] = img
        y_val[idx] = label

    # change text label(n04366367, ...) to (0, 1, 2, ...)
    print "encoding labels for validation data..."
    y_val = le.transform(y_val.tolist())

    return le, (X_train, y_train), (X_val, y_val)


def test_load_data():
    load_data('../../data/tiny-imagenet-200/')


if __name__ == '__main__':
    test_load_data()
