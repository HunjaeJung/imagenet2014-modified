import util
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import sparse_encode
from sklearn.svm import LinearSVC
import numpy as np


def selecting_non_zero_coef():
    """
    No End
    """
    (X_train, y_train), (X_test, y_test) = util.load_feat_vec()
    print "original X_train shape", X_train.shape
    X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(X_train, y_train)
    print "selected X_new shape", X_new.shape


def sparse_logisitic_regression():
    """
    If convolutional neural network features are highly overfitting.
    Then we could select features from sparse model.

    >> mean accuracy 0.352866
    """
    (X_train, y_train), (X_test, y_test) = util.load_feat_vec()

    clf = LogisticRegression(penalty='l1', multi_class='ovr')
    clf.fit(X_train, y_train)
    print "mean accuracy", clf.score(X_test, y_test)


def test_with_sparse_code(components=np.loadtxt('components_of_convfeat.txt')):
    (X_train, y_train), (X_test, y_test) = util.load_feat_vec()
    X_train_codes = np.loadtxt('sparse_codes_of_convfeat.txt')
    clf = LogisticRegression(penalty='l1', multi_class='ovr')
    clf.fit(X_train_codes, y_train)
    X_test_codes = sparse_encode(X_test, components)
    print "mean accuracy", clf.score(X_test_codes, y_test)


def learning_sparse_coding(X, components=None):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.sparse_encode.html
    """
    if components is None:
        print('Learning the dictionary...')
        t0 = time()
        diclearner = MiniBatchDictionaryLearning(n_components=100, verbose=True)
        components = diclearner.fit(X).components_
        np.savetxt('components_of_convfeat.txt', components)
        dt = time() - t0
        print('done in %.2fs.' % dt)

    codes = sparse_encode(X, components)
    np.savetxt('sparse_codes_of_convfeat.txt', codes)


if __name__ == '__main__':
    #sparse_logisitic_regression()
    #selecting_non_zero_coef()
    #(X_train, y_train), (X_test, y_test) = util.load_feat_vec()
    #learning_sparse_coding(X_train, components=np.loadtxt('components_of_convfeat.txt'))
    test_with_sparse_code()
