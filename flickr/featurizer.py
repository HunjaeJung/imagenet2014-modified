from PIL import Image
import leargist
import numpy as np
import pickle
import skimage.io
from skimage.feature import hog
import util
from skimage import data, color, exposure


def get_gist_features(data_path, pickle_name):
    size = len(data_path)
    gist_features = np.zeros((size, 960))
    for i in range(size):
        if i % 500 == 0: print "{}/{}".format(i, size)
        im = Image.open(data_path[i])
        gist_features[i, :] = leargist.color_gist(im)
    pickle.dump(gist_features, open(pickle_name, 'wb'), protocol=2)


def get_colorhistogram_features(data_path, pickle_name, binsize):
    size = len(data_path)
    nbin = 256/binsize
    hist_features = np.zeros((size, nbin*3))

    # global histogram
    for i in range(size):
        if i % 500 == 0: print "{}/{}".format(i, size)
        im = Image.open(data_path[i])
        nPixel = im.size[0] * im.size[1]
        hist = im.histogram()
        for ch in range(3):
            for n in range(nbin):
                hist_features[i, nbin*ch+n] = 1.0*sum(hist[ch*256+n*binsize:ch*256+(n+1)*binsize])/nPixel

    pickle.dump(hist_features, open(pickle_name, 'wb'), protocol=2)


def get_colorvariance_features(data_path, pickle_name):
    """
    Get Variance of Image Patch
    """
    size = len(data_path)
    rowPatchCnt = 4
    colPatchCnt = 4
    var_features = np.zeros((size, colPatchCnt*rowPatchCnt*3))
    print var_features.shape

    for i in range(size):
        if i % 500 == 0: print "{}/{}".format(i, size)
        im = util.load_image(data_path[i])
        patchH = im.shape[0] / rowPatchCnt
        patchW = im.shape[1] / colPatchCnt
        im = np.array(im)

        #print "***** ", i,"th image, shape : ",  im.shape, " *****"
        #print "patchW = ", patchW
        #print "patchH = ", patchH

        #print "{}'s im shape {}".format(i, im.shape)
        for w in range(rowPatchCnt):
            for h in range(rowPatchCnt):
                #print "feature idx : ", 3*(w*rowPatchCnt+h), 3*(w*rowPatchCnt+h+1)
                #print "input : ", np.std(im[h*patchH:(h+1)*patchH, w*patchW:(w+1)*patchW].reshape((patchW*patchH, 3)), axis=0)
                var_features[i, 3*(w*rowPatchCnt+h):3*(w*rowPatchCnt+h+1)] = np.std(im[h*patchH:(h+1)*patchH, w*patchW:(w+1)*patchW].reshape((patchW*patchH, 3)), axis=0)

    pickle.dump(var_features, open(pickle_name, 'wb'), protocol=2)


def get_HOG_features(data_path, pickle_name):
    size = len(data_path)
    rowPatchCnt = 4
    colPatchCnt = 4
    var_features = np.zeros((size, colPatchCnt*rowPatchCnt*3))
    print var_features.shape

    image = color.rgb2gray(data.astronaut())
    #print image

    fd, hog_image = hog(image, orientation = 8, pixels_per_cell=(16, 16), cells_per_block = (1,1), visualise=True)

    print fd

    im = util.load_image(data_path[0])
    #print im
    #for i in range(size):
        #if i % 500 == 0: print "{}/{}".format(i, size)
        #im = util.load_image(data_path[i])
        #patchH = im.shape[0] / rowPatchCnt
        #patchW = im.shape[1] / colPatchCnt
        #pass
        #im = np.array(im)

    pass


def get_SIFT_features(data_path, pickle_name):
    pass


if __name__ == '__main__':
    caffe_root = '../caffe/'
    flickr_train_set = np.loadtxt(caffe_root + 'data/flickr_style/train.txt', str, delimiter='\t')
    flickr_train_set = flickr_train_set[:63000]
    flickr_train_set_path = [readline.split()[0] for readline in flickr_train_set]

    flickr_test_set = np.loadtxt(caffe_root + 'data/flickr_style/test.txt', str, delimiter='\t')
    flickr_test_set = flickr_test_set[:15000]
    flickr_test_set_path = [readline.split()[0] for readline in flickr_test_set]

    #get_gist_features(flickr_train_set_path, 'feat_gist_train.pickle')
    #get_gist_features(flickr_test_set_path, 'feat_gist_test.pickle')
    #get_colorhistogram_features(flickr_train_set_path, 'feat_color_train24.pickle', binsize=32)
    #get_colorhistogram_features(flickr_test_set_path, 'feat_color_test24.pickle', binsize=32)
    get_HOG_features(flickr_train_set_path, 'feat_hog_train.pickle')
    get_HOG_features(flickr_test_set_path, 'feat_hog_test.pickle')
    #get_colorvariance_features(flickr_train_set_path, 'feat_var_train.pickle')
    #get_colorvariance_features(flickr_test_set_path, 'feat_var_test.pickle')
