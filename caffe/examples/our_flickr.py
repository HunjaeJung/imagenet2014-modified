from sklearn.svm import LinearSVC
import numpy as np
import caffe
import json
import gc
import pickle


caffe_root = '../'

MODEL_FILE_STYLE = '../models/Places_CNDS_models/deploy.prototxt'
PRETRAINED_STYLE = '../models/Places_CNDS_models/finetune_flickr_style_withplace_iter_5000.caffemodel'  # accuracy 0.38

"""
MODEL_FILE_STYLE = '../models/finetune_flickr_style/deploy.prototxt'
PRETRAINED_STYLE = '../models/finetune_flickr_style/finetune_flickr_style.caffemodel'
"""

MODEL_FILE_IMGNET = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED_IMGNET = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

style_dic = np.loadtxt(caffe_root + 'examples/finetune_flickr_style/style_names.txt', str, delimiter='\t')
synset_dic = np.loadtxt(caffe_root + 'data/flickr_style/synset_words.txt', str, delimiter='\t')
flickr_test_set = np.loadtxt(caffe_root + 'data/flickr_style/test.txt', str, delimiter='\t')
flickr_test_set_path = [readline.split()[0] for readline in flickr_test_set]
flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]
flickr_train_set = np.loadtxt(caffe_root + 'data/flickr_style/train.txt', str, delimiter='\t')
flickr_train_set_path = [readline.split()[0] for readline in flickr_train_set]
flickr_train_set_label = [int(readline.split()[1]) for readline in flickr_train_set]


class OurFlickr():
    def __init__(self):
        self.net = caffe.Classifier(MODEL_FILE_STYLE, PRETRAINED_STYLE,
                                    mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                                    channel_swap=(2, 1, 0),
                                    raw_scale=255,
                                    image_dims=(227, 227))
        print "wow"
        self.net_obj = caffe.Classifier(MODEL_FILE_IMGNET, PRETRAINED_IMGNET,
                                        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                                        channel_swap=(2, 1, 0),
                                        raw_scale=255,
                                        image_dims=(256, 256))

    def fit(self, X_paths, y_paths):
        """
        Our fitting
        """
        ntrain = len(X_paths)
        batch = 1000
        nbatch = ntrain/batch
        feat_vec = np.zeros((nbatch*batch, 4096))
        our_model = OurFlickr()

        for n in range(nbatch):
            X_batch, _ = get_train_dataset(flickr_train_set_path[batch*n:batch*(n+1)], flickr_train_set_label[batch*n:batch*(n+1)])
            featvec_batch = our_model.net_extractor(X=X_batch)
            feat_vec[batch*n:batch*(n+1)] = featvec_batch
            gc.collect()

        #self.X = X
        #self.y = y
        #self.nsamples = len(X)

    def transform(self):
        print "transform start"
        feat_conv = self.net_extractor()
        feat_hand = self.handcraft_extractor()
        self.X_new = np.hstack((feat_conv, feat_hand))
        print "transform end"

    def compile(self):
        print "compile start"
        self.clf = LinearSVC()
        self.clf.fit(self.X_new, self.y)
        print "compile end"

    def _fit(self, y, pickle_name=None):
        feat_conv = pickle.load(open(pickle_name, 'rb'))
        feat_hand = self.handcraft_extractor(X=np.zeros((63000, 1)))
        self.X_new = np.hstack((feat_conv, feat_hand))
        self.y = y

    def predict_our(self, img):
        """
        predict only one instance
        """
        predict = self.net.predict([img], oversample=True)
        test_feat_conv = self.net_extractor([img], oversample=True)
        test_feat_hand = self.handcraft_extractor(X=np.array([img]))

        test_feat = np.hstack((test_feat_conv, test_feat_hand))
        print "convnet predict: ", predict[0].argsort()[-1:-6:-1]
        y_sfmax = predict[0].argsort()[-1]
        y_svm = self.clf.predict(test_feat)
        return y_sfmax, y_svm[0]

    def net_extractor(self, X=None, oversample=False):
        """
        caffe runs on batch size 10, it needs to iterate by 10.
        oversampling expands dataset
        """
        if X is None:
            X = self.X

        feat_conv = np.zeros((len(X), 4096))

        if not oversample and len(X) >= 10:
            nbatch = len(X)/10
            for n in range(nbatch):
                print "net extract batch : {}, oversample: {}".format(n, oversample)
                self.net.predict(X[n*10:(n+1)*10], oversample=oversample)
                feat_conv[n*10:(n+1)*10, :] = self.net.blobs['fc7'].data
        else:
            for n in range(len(X)):
                print "net extract batch : {}, oversample: {}".format(n, oversample)
                self.net.predict([X[n]], oversample=oversample)
                feat_conv[n] = np.mean(self.net.blobs['fc7'].data, axis=0)
        return feat_conv

    def handcraft_extractor(self, X=None):
        if X is None:
            X = self.X
        nfeatures = 3
        feat_hand = np.ones((len(X), nfeatures))
        return feat_hand

    def predict(self, img):
        """
        TODO very very dirty code.. plz fixme after text..
        """
        predict_style = self.net.predict([img], oversample=True)
        predict_obj = self.net_obj.predict([img], oversample=True)
        #y_sfmax, y_svm = self.predict_our(img)

        predict_style_top5 = predict_style[0].argsort()[-1:-6:-1]
        predict_obj_top5 = predict_obj[0].argsort()[-1:-6:-1]

        res_dic = {'predict_style': [{}]*5,
                   'predict_obj': [{}]*5,
                   #'predict_our': {'name': style_dic[y_svm]}
                   }

        for i, label in enumerate(predict_style_top5):
            print "Top{0}: {1} - {2}".format(i, style_dic[label], predict_style[0][label])

            res_dic['predict_style'][i] = {'name': style_dic[label],
                                           'score': str(predict_style[0][label])}

        for i, label in enumerate(predict_obj_top5):
            print "Top(syns){0}: {1} - {2}".format(i, synset_dic[label], predict_obj[0][label])
            res_dic['predict_obj'][i] = {'label': synset_dic[label].split()[0],
                                         'score': str(predict_obj[0][label]),
                                         'name': ''.join(synset_dic[label].split()[1:])}
        return res_dic

    def predict_style(self, img):
        predict_style = self.net.predict([img], oversample=True)
        return predict_style[0].argsort()[-1:-6:-1]

    def predict_obj_top5(self, img):
        predict_obj = self.net_obj.predict([img], oversample=True)
        predict_obj_top5 = predict_obj[0].argsort()[-1:-6:-1]
        predict_confidence = []

        for label in predict_obj_top5:
            predict_confidence.append(predict_obj[0][label])

        return predict_obj_top5, predict_confidence


def get_train_dataset(paths, labels):
    print "get datasets"
    X = list()
    cnt = 0
    for path, label in zip(paths, labels):
        if cnt % 1000 == 0: gc.collect()
        X.append(caffe.io.load_image(path))
        cnt += 1
        if cnt % 500 == 0: print cnt
    return X, labels


def test_predict():
    X_train, y_train = get_train_dataset(flickr_train_set_path[:100], flickr_train_set_label[:100])
    img = caffe.io.load_image(flickr_test_set_path[9])

    our_model = OurFlickr()
    our_model.fit(X_train, y_train)  # TODO SVM batch fitting
    our_model.transform()
    our_model.compile()

    res = our_model.predict(img)


    print json.dumps(res)


def obj_labeler():
    import os
    dir_path = '/shared/flickr_style/images/'

    print "hello"
    caffe.set_mode_cpu()
    our_model = OurFlickr()
    print "hello"

    with open('/shared/flickr_style/flickr_obj_label.txt', 'w') as f:
        for filename in os.listdir(dir_path):
            img = caffe.io.load_image(dir_path + filename)

            predict_obj_top5, confidence = our_model.predict_obj_top5(img)
            classes = [synset_dic[idx].split()[0] for idx in predict_obj_top5]

            res = [str(j) for i in zip(classes, confidence) for j in i]
            print filename, res
            f.write(','.join([dir_path+filename] + res) + '\n')

def style_labeler():
    ntrain = 70000  # max is 7000, no 7500
    X_train, y_train = get_train_dataset(flickr_train_set_path[:ntrain], flickr_train_set_label[:ntrain])

    caffe.set_mode_gpu()
    our_model = OurFlickr()
    our_model.fit(X_train, y_train)  # TODO SVM batch fitting
    our_model.transform()
    our_model.compile()

    true_res = []
    svm_res = []
    sfmax_res = []
    with open('./label_result_all.csv', 'w') as f:
        for i in range(len(flickr_test_set_path)):
            if i % 1000: gc.collect()
            print i
            img = caffe.io.load_image(flickr_test_set_path[i])
            sfmax, svm = our_model.predict_our(img)
            sfmax_res.append(sfmax)
            svm_res.append(svm)
            true_res.append(flickr_test_set_label[i])
            f.write(",".join([flickr_test_set_path[i],
                              str(flickr_test_set_label[i]),
                              str(sfmax),
                              str(svm[0])]) + "\n")

    print "svm accuarcy:", np.mean([a == b for a, b in zip(true_res, svm_res)])
    print "sfmax accuracy:", np.mean([a == b for a, b in zip(true_res, sfmax_res)])

def make_train_featvec(X, y):
    """
    This methods needs sufficient memories
    """
    ntrain = len(X)
    batch = 1000
    nbatch = ntrain/batch
    feat_vec = np.zeros((nbatch*batch, 4096))

    caffe.set_mode_gpu()
    our_model = OurFlickr()

    for n in range(nbatch):
        print "## batch {}/{} ##".format(n, nbatch)
        X_batch, _ = get_train_dataset(X[batch*n:batch*(n+1)], y[batch*n:batch*(n+1)])
        featvec_batch = our_model.net_extractor(X=X_batch)
        feat_vec[batch*n:batch*(n+1)] = featvec_batch
        gc.collect()

    del our_model
    del X_batch
    del X
    del y
    gc.collect()

    print feat_vec
    print feat_vec.shape
    pickle.dump(feat_vec, open('feat_test_place.pickle', 'wb'), protocol=2)

def main():
    ntrain = 4000  # max is 7000, no 7500
    #itest = 9

    caffe.set_mode_gpu()

    our_model = OurFlickr()
    #### past fail model####
    #X_train, y_train = get_train_dataset(flickr_train_set_path[:ntrain], flickr_train_set_label[:ntrain])
    #our_model.fit(X_train, y_train)  # TODO SVM batch fitting
    #our_model.transform()
    #our_model.compile()

    #### use pickled training data features ####
    our_model._fit(flickr_train_set_label[:63000], pickle_name='feat_train.pickle')
    our_model.compile()

    true_res = []
    svm_res = []
    sfmax_res = []
    for i in range(15000):
        img = caffe.io.load_image(flickr_test_set_path[i])
        sfmax, svm = our_model.predict_our(img)
        sfmax_res.append(sfmax)
        svm_res.append(svm)
        true_res.append(flickr_test_set_label[i])
        print "true label", flickr_test_set_label[i]

    print "svm accuarcy:", np.mean([a == b for a, b in zip(true_res, svm_res)])
    print "sfmax accuracy:", np.mean([a == b for a, b in zip(true_res, sfmax_res)])

if __name__ == '__main__':
    #main()
    make_train_featvec(flickr_test_set_path, flickr_test_set_label)
    #make_train_featvec(flickr_train_set_path, flickr_train_set_label)
    #print len(flickr_train_set_path)
    #test_predict()
    #obj_labeler()
    #style_labeler()
