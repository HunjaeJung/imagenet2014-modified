import sys
sys.path.append('../keras')

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

from PIL import Image
import leargist
import numpy as np
import caffe
import gc

caffe_root = '../caffe/'
project_root = '../'
style_dic = np.loadtxt(caffe_root + 'examples/finetune_flickr_style/style_names.txt', str, delimiter='\t')
synset_dic = np.loadtxt(caffe_root + 'data/flickr_style/synset_words.txt', str, delimiter='\t')


class BongguNet():
    def __init__(self,
                 clf=None,
                 style_deploy=caffe_root+'models/finetune_flickr_style/deploy.prototxt',
                 style_pretrain=caffe_root+'models/finetune_flickr_style/finetune_flickr_style.caffemodel',
                 obj_deploy=caffe_root+'models/bvlc_reference_caffenet/deploy.prototxt',
                 obj_pretrain=caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                 clf_hdf5=project_root+'flickr/our_final_classifier.hdf5',
                 input_ndim=5200):

        # conv feature generator
        self.net = caffe.Classifier(style_deploy, style_pretrain,
                                    mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                                    channel_swap=(2, 1, 0),
                                    raw_scale=255,
                                    image_dims=(227, 227))

        self.net_obj = caffe.Classifier(obj_deploy, obj_pretrain,
                                        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                                        channel_swap=(2, 1, 0),
                                        raw_scale=255,
                                        image_dims=(256, 256))
        # gist feature generator

        # final classifier
        if clf is None:
            opt = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=False)
            self.clf = classifier(input_ndim)
            self.clf.load_weights(clf_hdf5)
            self.clf.compile(loss='categorical_crossentropy', optimizer=opt)

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

    def handcraft_extractor(self, img):
        feat_gist = self.get_gist(img)
        feat_colorhist = self.get_colorhist(img)
        feat_colorvar = self.get_colorvar(img)

        feat_hand = np.hstack((feat_gist, feat_colorhist, feat_colorvar))
        return feat_hand

    def get_gist(self, img):
        gist_features = np.zeros((1, 960))
        im = Image.fromarray(img.astype('uint8'))
        gist_features[0, :] = leargist.color_gist(im)
        return gist_features

    def get_colorhist(self, img):
        binsize = 8
        im = Image.fromarray(img.astype('uint8'))
        nbin = 256/binsize
        nPixel = im.size[0] * im.size[1]
        hist = im.histogram()

        hist_features = np.zeros((1, nbin*3))

        for ch in range(3):
            for n in range(nbin):
                hist_features[0, nbin*ch+n] = 1.0*sum(hist[ch*256+n*binsize:ch*256+(n+1)*binsize])/nPixel
        return hist_features

    def get_colorvar(self, img):
        rowPatchCnt = 4
        colPatchCnt = 4
        patchH = img.shape[0] / rowPatchCnt
        patchW = img.shape[1] / colPatchCnt

        var_features = np.zeros((1, colPatchCnt*rowPatchCnt*3))

        for w in range(rowPatchCnt):
            for h in range(rowPatchCnt):
                var_features[0, 3*(w*rowPatchCnt+h):3*(w*rowPatchCnt+h+1)] = np.std(img[h*patchH:(h+1)*patchH, w*patchW:(w+1)*patchW].reshape((patchW*patchH, 3)), axis=0)
        return var_features

    def predict(self, img):
        feat_convNN = self.net_extractor([img], oversample=True)
        feat_handcraft = self.handcraft_extractor(img)
        X_test = np.hstack((feat_convNN, feat_handcraft))

        predict_style = self.clf.predict(X_test)
        predict_obj = self.net_obj.predict([img], oversample=True)

        predict_style_top5 = predict_style[0].argsort()[-1:-6:-1]
        predict_obj_top5 = predict_obj[0].argsort()[-1:-6:-1]

        res_dic = {'predict_style': [{}]*5,
                   'predict_obj': [{}]*5}

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

    def predict_our(self, img):
        feat_convNN = self.net_extractor([img], oversample=True)
        feat_handcraft = self.handcraft_extractor(img)
        X_test = np.hstack((feat_convNN, feat_handcraft))

        predict_style = self.clf.predict(X_test)
        return predict_style[0].argsort()[-1:-6:-1]


def classifier(input_dim):
    model = Sequential()
    model.add(Dense(input_dim, 2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    model.add(Dense(2048, 20))
    model.add(Activation('softmax'))
    return model


def test_predict():
    flickr_test_set = np.loadtxt(caffe_root + 'data/flickr_style/test.txt', str, delimiter='\t')
    flickr_test_set_path = [readline.split()[0] for readline in flickr_test_set]
    flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]

    n = 9
    img = caffe.io.load_image(flickr_test_set_path[n])
    print "true label", flickr_test_set_label[n]

    net = BongguNet()
    res = net.predict(img)
    print res


def style_labeler():
    flickr_test_set = np.loadtxt(caffe_root + 'data/flickr_style/test.txt', str, delimiter='\t')
    flickr_test_set_path = [readline.split()[0] for readline in flickr_test_set]
    flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]

    caffe.set_mode_gpu()
    our_model = BongguNet()

    true_res = []
    our_res = []
    our_res5 = []

    with open('./label_result_bonggunet.csv', 'w') as f:
        for i in range(len(flickr_test_set_path)):
            if i % 1000: gc.collect()
            img = caffe.io.load_image(flickr_test_set_path[i])
            res = our_model.predict_our(img)

            our_res.append(res[0])
            our_res5.append(res)
            true_res.append(flickr_test_set_label[i])
            print our_res, true_res
            f.write(",".join([str(flickr_test_set_label[i]),
                              flickr_test_set_path[i],
                              str(true_res[i]),
                              str(our_res[i])]) + "\n")

    print "accuarcy@1:", np.mean([a == b for a, b in zip(true_res, our_res)])
    print "accuarcy@5:", np.mean([a in b for a, b in zip(true_res, our_res5)])


if __name__ == '__main__':
    #test_predict()
    style_labeler()
