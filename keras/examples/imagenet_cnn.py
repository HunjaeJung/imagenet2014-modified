# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from keras.datasets import tinyimagenet
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils, noti_utils, log_utils
from six.moves import range
import numpy as np
import time

'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imagenet_cnn.py

    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).

    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''

batch_size = 32
nb_classes = 200
nb_epoch = 20
data_augmentation = False

# the data, shuffled and split between tran and test sets
label_encoder, (X_train, y_train), (X_test, y_test) = tinyimagenet.load_data('/shared/tiny-imagenet-200/')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

nkerns = [3, 32, 32, 64, 64]
nweigts = 0
act_func = 'tanh'

# (32, 3, 3, 3) only define kernel(filter) size
model.add(Convolution2D(nkerns[1], nkerns[0], 3, 3, border_mode='full'))
model.add(Activation(act_func))
model.add(Convolution2D(nkerns[2], nkerns[1], 3, 3))
model.add(Activation(act_func))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(nkerns[3], nkerns[2], 3, 3, border_mode='full'))
model.add(Activation(act_func))
model.add(Convolution2D(nkerns[4], nkerns[3], 3, 3))
model.add(Activation(act_func))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(nkerns[4]*16*16, 512, init='normal'))
model.add(Activation(act_func))
model.add(Dropout(0.5))

model.add(Dense(512, nb_classes, init='normal'))
model.add(Activation('softmax'))

try:
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    if not data_augmentation:
        print("Not using data augmentation or normalization")
        noti_utils.notify('Start')
        start_time = time.time()
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")
        X_train /= 255
        X_test /= 255
        history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
        print history
        score = model.evaluate(X_test, Y_test, batch_size=batch_size)
        classes = model.predict_classes(X_test, batch_size=batch_size)
        acc = np_utils.accuracy(classes, y_test)
        label_result = label_encoder.inverse_transform(classes)
        running_time = time.time() - start_time

        data_config =  ['tiny imageNet',
                        len(X_train),   # num of training data
                        'tiny-imagenet-200/val',
                        len(X_test),    # num of test set
                        ]

        network_config = [act_func,    # activation funciton
                          4,              # num of conv layer
                          2,              # num of max pooling
                          batch_size,     # batch size
                          10000000000,    # num of weights
                          10              # epoch
                          ]

        exp_result = [score, acc, running_time]

        log_utils.write_log(data_config, exp_result, network_config, history, label_result)
        noti_utils.notify('Done > score : ', score ,', accuracy : ', acc )

        print('Test score:', score)
        print('Test accuracy:', acc)
        print label_result
    else:
        print("Using real time data augmentation")

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=True, # set input mean to 0 over the dataset
            samplewise_center=False, # set each sample mean to 0
            featurewise_std_normalization=True, # divide inputs by std of the dataset
            samplewise_std_normalization=False, # divide each input by its std
            zca_whitening=False, # apply ZCA whitening
            rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
            horizontal_flip=True, # randomly flip images
            vertical_flip=False) # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        for e in range(nb_epoch):
            print('-'*40)
            print('Epoch', e)
            print('-'*40)
            print("Training...")
            # batch train with realtime data augmentation
            progbar = generic_utils.Progbar(X_train.shape[0])
            for X_batch, Y_batch in datagen.flow(X_train, Y_train):
                loss = model.train(X_batch, Y_batch)
                progbar.add(X_batch.shape[0], values=[("train loss", loss)])

            print("Testing...")
            # test time!
            progbar = generic_utils.Progbar(X_test.shape[0])
            for X_batch, Y_batch in datagen.flow(X_test, Y_test):
                score = model.test(X_batch, Y_batch)
                progbar.add(X_batch.shape[0], values=[("test loss", score)])
except Exception as e:
    noti_utils.notify("There are some error!, Try FIx it now", e)

