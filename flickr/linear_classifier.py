# -*- coding: utf-8 -*-

import util
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def classifiy_svm():
    print "SVM"
    (X_train, y_train), (X_test, y_test) = util.load_all_feat()
    print "original X_train shape", X_train.shape
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print "accuracy score:", accuracy_score(y_test, pred)


def classify_perceptron():
    print "perceptron"
    (X_train, y_train), (X_test, y_test) = util.load_all_feat()
    print "original X_train shape", X_train.shape
    clf = Perceptron()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print "accuracy score:", accuracy_score(y_test, pred)


def classify_logistic():
    print "logistic regression"
    (X_train, y_train), (X_test, y_test) = util.load_all_feat()
    print "original X_train shape", X_train.shape
    clf = RandomizedLogisticRegression(n_jobs=2)
    clf.fit(X_train, y_train)
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print "accuracy score:", accuracy_score(y_test, pred)

    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT



def classify_sgd(loss="hinge"):
    print "SGD Clasifier with loss function({})".format(loss)
    (X_train, y_train), (X_test, y_test) = util.load_all_feat()
    X_train = X_train[:, :4096]
    X_test = X_test[:, :4096]
    clf = SGDClassifier(loss=loss, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print "accuracy score:", accuracy_score(y_test, pred)

    nclass = len(clf.coef_)

    #plt.figure()
    #colors = ['r']*4096 + ['b']*960 + ['g']*96 + ['m']*48
    #for n in range(nclass):
        #plt.subplot(4, 5, n+1)  # 4 5
        #for p in range(len(clf.coef_[n])):
            #plt.bar(p, clf.coef_[n][p], color=colors[p])
    #plt.savefig('~/Dropbox/linux/vision_project/img_new/weight_score_{}.png'.format(loss))


if __name__ == '__main__':
    #classify_perceptron()
    #classify_logistic()
    #classifiy_svm()
    classify_sgd(loss="squared_hinge")
