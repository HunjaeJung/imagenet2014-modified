import sys
sys.path.append('../keras')

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.regularizers import l2, l1, l1l2
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils


def a():
    # our final classifier
    opt = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=False)
    clf = create_model(X_train.shape[1])
    clf.load_weights('./our_final_classifier.hdf5')
    clf.compile(loss='categorical_crossentropy', optimizer=opt)

    # final predict
    score = clf.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def classifier(input_dim):
    model = Sequential()
    model.add(Dense(input_dim, 2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    model.add(Dense(2048, 20))
    model.add(Activation('softmax'))
    return model
