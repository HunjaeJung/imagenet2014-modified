# -*- coding: utf-8 -*-

import sys
sys.path.append('../keras')

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.regularizers import l2, l1, l1l2
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils

import numpy as np
import pickle
import util

'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).

    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''

caffe_root = '../caffe/'
flickr_test_set = np.loadtxt(caffe_root + 'data/flickr_style/test.txt', str, delimiter='\t')
flickr_test_set = flickr_test_set[:15000]
flickr_test_set_path = [readline.split()[0] for readline in flickr_test_set]
flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]
flickr_train_set = np.loadtxt(caffe_root + 'data/flickr_style/train.txt', str, delimiter='\t')
flickr_train_set = flickr_train_set[:63000]
flickr_train_set_path = [readline.split()[0] for readline in flickr_train_set]
flickr_train_set_label = [int(readline.split()[1]) for readline in flickr_train_set]


def NN_only_convfeat():
    batch_size = 64
    nb_classes = 20
    nb_epoch = 20

    np.random.seed(1337)  # for reproducibility

    print "let's load"
    X_train = pickle.load(open('feat_train.pickle', 'rb'))
    X_test = pickle.load(open('feat_test.pickle', 'rb'))
    Y_train = np_utils.to_categorical(flickr_train_set_label, nb_classes)
    Y_test = np_utils.to_categorical(flickr_test_set_label, nb_classes)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    model = Sequential()
    model.add(Dense(4096, 2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2048, 1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, 20))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    print "model compile end"

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_split=0.2)
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def NN_with_allfeat(epoch=20):
    batch_size = 64
    nb_classes = 20
    nb_epoch = epoch

    np.random.seed(1337)  # for reproducibility

    print "let's load"
    X_train_convfeat = pickle.load(open('feat_train.pickle', 'rb'))
    X_train_gistfeat = pickle.load(open('feat_gist_train.pickle', 'rb'))
    X_train_colorfeat = pickle.load(open('feat_color_train.pickle', 'rb'))
    X_train_varfeat = pickle.load(open('feat_var_train.pickle', 'rb'))

    X_test_convfeat = pickle.load(open('feat_test.pickle', 'rb'))
    X_test_gistfeat = pickle.load(open('feat_gist_test.pickle', 'rb'))
    X_test_colorfeat = pickle.load(open('feat_color_test.pickle', 'rb'))
    X_test_varfeat = pickle.load(open('feat_var_test.pickle', 'rb'))

    # acc@1 0.3945
    #X_train = np.hstack((X_train_convfeat, X_train_gistfeat, X_train_colorfeat, X_train_varfeat))
    #X_test = np.hstack((X_test_convfeat, X_test_gistfeat, X_test_colorfeat, X_test_varfeat))

    # acc@1 0.3942
    #X_train = X_train_convfeat
    #X_test = X_test_convfeat

    # acc@1 0.3949
    #X_train = np.hstack((X_train_convfeat, X_train_gistfeat))
    #X_test = np.hstack((X_test_convfeat, X_test_gistfeat))

    # acc@1 0.3941
    X_train = np.hstack((X_train_convfeat, X_train_gistfeat, X_train_colorfeat))
    X_test = np.hstack((X_test_convfeat, X_test_gistfeat, X_test_colorfeat))

    print "X_shape", X_train_convfeat.shape[1], X_train_gistfeat.shape[1], X_train_colorfeat.shape[1], X_train_varfeat.shape[1]
    Y_train = np_utils.to_categorical(flickr_train_set_label, nb_classes)
    Y_test = np_utils.to_categorical(flickr_test_set_label, nb_classes)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    """
    model = Sequential()
    model.add(Dense(X_train.shape[1], 2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    model.add(Dense(2048, 1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    model.add(Dense(1024, 20))
    model.add(Activation('softmax'))
    """

    model = Sequential()
    model.add(Dense(X_train.shape[1], 2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    model.add(Dense(2048, 20))
    model.add(Activation('softmax'))

    #opt = Adagrad(lr=0.01, epsilon=1e-6)
    #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, kappa=1-1e-8)
    #opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    opt = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    print "model compile end"

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_split=0.2)
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights('./model_weights.hdf5')

    model_loaded = create_model(X_train.shape[1])
    model_loaded.load_weights('./model_weights.hdf5')

    model_loaded.compile(loss='categorical_crossentropy', optimizer=opt)
    score = model_loaded.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def create_model(input_dim):
    model = Sequential()
    model.add(Dense(input_dim, 2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    model.add(Dense(2048, 20))
    model.add(Activation('softmax'))
    return model


def NN_with_join(epoch=20):
    """
    Sadly, cannot merge
    """
    batch_size = 64
    nb_classes = 20
    nb_epoch = epoch

    np.random.seed(1337)  # for reproducibility
    X_train_convfeat = pickle.load(open('feat_train.pickle', 'rb'))
    X_train_gistfeat = pickle.load(open('feat_gist_train.pickle', 'rb'))

    X_test_convfeat = pickle.load(open('feat_test.pickle', 'rb'))
    X_test_gistfeat = pickle.load(open('feat_gist_test.pickle', 'rb'))

    Y_train = np_utils.to_categorical(flickr_train_set_label, nb_classes)
    Y_test = np_utils.to_categorical(flickr_test_set_label, nb_classes)

    left = Sequential()
    print "X_train_convfeat.shape[1]", X_train_convfeat.shape[1]
    left.add(Dense(X_train_convfeat.shape[1], 50))
    left.add(Activation('relu'))

    right = Sequential()
    print "X_train_gistfeat.shape[1]", X_train_gistfeat.shape[1]
    right.add(Dense(X_train_gistfeat.shape[1], 50))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))

    model.add(Dense(50, 10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train_convfeat, X_train_gistfeat], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([X_test_convfeat, X_test_gistfeat], Y_test))

if __name__ == '__main__':
    NN_with_allfeat(epoch=20)
    #NN_only_convfeat()
    #NN_with_join(epoch=20)
