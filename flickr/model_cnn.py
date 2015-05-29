# -*- coding: utf-8 -*-

import sys
sys.path.append('../keras')

#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
import numpy as np
import util
import gc


def batch_generator(file_pathes, batch_size=252, max_iter=1):
    nbatch = len(file_pathes)/batch_size
    for n in range(max_iter):
        for idx in range(nbatch):
            X = np.zeros((batch_size, 3, 256, 256))

            # [idx*batch_size:(idx+1)*batch_size]
            for i, path in enumerate(file_pathes[idx*batch_size:(idx+1)*batch_size]):
                im = np.array(util.load_image(path))
                img = np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]])
                X[i, :, :, :] = img

            yield X


def our_model():
    """
    In GTX980, maximum training batch size is 40
    Train_batch_size + test_batch_size < 9000
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python model_cnn.py &> cnn.log

    fail case:
    train_batch_size = 7000
    test_batch_size = 1500
    train_gpu_batch_size = 40  # it should divide train_batch_size
    test_gpu_batch_size = 40

    In this case, train phase, 5000 MB occupied:
    train_batch_size = 3000
    test_batch_size = 3000
    train_gpu_batch_size = 40  # it should divide train_batch_size
    test_gpu_batch_size = 40
    """
    # Parameters
    train_batch_size = 3000
    test_batch_size = 3000
    train_gpu_batch_size = 40  # it should divide train_batch_size
    test_gpu_batch_size = 40

    ntrain = 63000
    ntest = 15000
    nb_epoch = 10

    # Data Load
    flickr_train_set = np.loadtxt('/shared/flickr_style/train_resized.txt', str, delimiter='\t')
    flickr_train_set = flickr_train_set[:ntrain]
    flickr_train_set_path = [readline.split()[0] for readline in flickr_train_set]
    flickr_train_set_label = [int(readline.split()[1]) for readline in flickr_train_set]

    flickr_test_set = np.loadtxt('/shared/flickr_style/test_resized.txt', str, delimiter='\t')
    flickr_test_set = flickr_test_set[:ntest]
    flickr_test_set_path = [readline.split()[0] for readline in flickr_test_set]
    flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]

    Y_train = np_utils.to_categorical(flickr_train_set_label, 20)
    Y_test = np_utils.to_categorical(flickr_test_set_label, 20)

    # Model Define
    print "Model Compile Start..."
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 32, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64*64*64, 512, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, 20, init='normal'))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print "... done"

    for e in range(nb_epoch):
        """
        # Read by Read
        print("Training...")
        for idx, X_train_batch in enumerate(batch_generator(flickr_train_set_path, batch_size=train_batch_size, max_iter=1)):
            Y_train_batch = Y_train[idx*train_batch_size:(idx+1)*train_batch_size]
            tr_loss, tr_accuracy = model.train(X_train_batch, Y_train_batch, accuracy=True)
            #print "epoch,", e, "batch,", idx, "train loss,", tr_loss, "train accraucy@1", tr_accuracy

        print("Testing...")
        for idx, X_test_batch in enumerate(batch_generator(flickr_test_set_path, batch_size=test_batch_size, max_iter=1)):
            Y_test_batch = Y_test[idx*test_batch_size:(idx+1)*test_batch_size]
            test_loss, test_accuracy = model.test(X_test_batch, Y_test_batch)
            print "epoch,", e, "batch,", idx, "test loss,", test_loss, "test accraucy@1", test_accuracy
        """

        # Big Batch in Memory, small batch in GPU Memory
        for idx, X_train_batch in enumerate(batch_generator(flickr_train_set_path, batch_size=train_batch_size, max_iter=1)):
            Y_train_batch = Y_train[idx*train_batch_size:(idx+1)*train_batch_size]
            for gpuidx in range(train_batch_size/train_gpu_batch_size):
                #X_train_batch_gpu = X_train_batch[gpuidx*train_gpu_batch_size:(gpuidx+1)*train_gpu_batch_size]
                #Y_train_batch_gpu = Y_train_batch[gpuidx*train_gpu_batch_size:(gpuidx+1)*train_gpu_batch_size]
                loss, accuracy = model.train(X_train_batch[gpuidx*train_gpu_batch_size:(gpuidx+1)*train_gpu_batch_size], Y_train_batch[gpuidx*train_gpu_batch_size:(gpuidx+1)*train_gpu_batch_size], accuracy=True)
                print "train,epoch,{},batch,{},gpubatch,{},trainloss,{},trainaccruacy@1,{}".format(e, idx, gpuidx, loss, accuracy)
            gc.collect()
        del X_train_batch
        del Y_train_batch

        for idx, X_test_batch in enumerate(batch_generator(flickr_test_set_path, batch_size=test_batch_size, max_iter=1)):
            Y_test_batch = Y_test[idx*test_batch_size:(idx+1)*test_batch_size]
            for gpuidx in range(test_batch_size/test_gpu_batch_size):
                #X_test_batch_gpu = X_test_batch[gpuidx*test_gpu_batch_size:(gpuidx+1)*test_gpu_batch_size]
                #Y_test_batch_gpu = Y_test_batch[gpuidx*test_gpu_batch_size:(gpuidx+1)*test_gpu_batch_size]
                loss, accuracy = model.test(X_test_batch[gpuidx*test_gpu_batch_size:(gpuidx+1)*test_gpu_batch_size], Y_test_batch[gpuidx*test_gpu_batch_size:(gpuidx+1)*test_gpu_batch_size], accuracy=True)
                print "test,epoch,{},batch,{},gpubatch,{},testloss,{},testaccruacy@1,{}".format(e, idx, gpuidx, loss, accuracy)
            gc.collect()
        del X_test_batch
        del Y_test_batch

if __name__ == '__main__':
    our_model()
