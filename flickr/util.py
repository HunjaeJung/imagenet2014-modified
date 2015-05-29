import numpy as np
import pickle
import sys
sys.path.append('../keras')
from keras.utils import np_utils
import skimage.io
import gc


def make_batch_pickle(file_pathes, batch_size=63, rootpath='/shared/flickr_style/train/'):
    """
    pickle is really big. so we couldn't use this method
    """
    cnt = 0
    cnt_batch = 0
    total_batch = len(file_pathes)/batch_size

    for path in file_pathes:
        X_batch = np.zeros((batch_size, 3, 256, 256))
        if cnt < batch_size:
            print "progress.. {}/{}".format(cnt, batch_size)
            im = np.array(load_image(path))
            img = np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]])
            X_batch[cnt, :, :, :] = img
            cnt += 1
        elif cnt == batch_size:
            pickle.dump(X_batch, open(rootpath + 'flickr_train{}.pickle'.format(cnt_batch), 'wb'))
            cnt = 0
            cnt_batch += 1
            print "batch {}/{}".format(cnt_batch, total_batch)
            gc.collect()


def load_feat_vec(class_type=None):
    caffe_root = '../caffe/'
    flickr_test_set = np.loadtxt(caffe_root + 'data/flickr_style/test.txt', str, delimiter='\t')
    flickr_test_set = flickr_test_set[:15000]
    flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]
    flickr_train_set = np.loadtxt(caffe_root + 'data/flickr_style/train.txt', str, delimiter='\t')
    flickr_train_set = flickr_train_set[:63000]
    flickr_train_set_label = [int(readline.split()[1]) for readline in flickr_train_set]

    X_train = pickle.load(open('feat_train.pickle', 'rb'))
    X_test = pickle.load(open('feat_test.pickle', 'rb'))
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    if class_type == 'categorical':
        Y_train = np_utils.to_categorical(flickr_train_set_label, 20)
        Y_test = np_utils.to_categorical(flickr_test_set_label, 20)

        return (X_train, Y_train), (X_test, Y_test)
    else:
        y_train = flickr_train_set_label
        y_test = flickr_test_set_label

        return (X_train, y_train), (X_test, y_test)


def load_all_feat(class_type=None):
    caffe_root = '../caffe/'
    flickr_test_set = np.loadtxt(caffe_root + 'data/flickr_style/test.txt', str, delimiter='\t')
    flickr_test_set = flickr_test_set[:15000]
    flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]
    flickr_train_set = np.loadtxt(caffe_root + 'data/flickr_style/train.txt', str, delimiter='\t')
    flickr_train_set = flickr_train_set[:63000]
    flickr_train_set_label = [int(readline.split()[1]) for readline in flickr_train_set]

    X_train_convfeat = pickle.load(open('feat_train.pickle', 'rb'))
    X_train_gistfeat = pickle.load(open('feat_gist_train.pickle', 'rb'))
    X_train_colorfeat = pickle.load(open('feat_color_train.pickle', 'rb'))
    X_train_varfeat = pickle.load(open('feat_var_train.pickle', 'rb'))

    X_test_convfeat = pickle.load(open('feat_test.pickle', 'rb'))
    X_test_gistfeat = pickle.load(open('feat_gist_test.pickle', 'rb'))
    X_test_colorfeat = pickle.load(open('feat_color_test.pickle', 'rb'))
    X_test_varfeat = pickle.load(open('feat_var_test.pickle', 'rb'))

    print X_train_convfeat.shape, X_train_gistfeat.shape, X_train_colorfeat.shape, X_train_varfeat.shape

    X_train = np.hstack((X_train_convfeat, X_train_gistfeat, X_train_colorfeat, X_train_varfeat))
    X_test = np.hstack((X_test_convfeat, X_test_gistfeat, X_test_colorfeat, X_test_varfeat))

    Y_train = np_utils.to_categorical(flickr_train_set_label, 20)
    Y_test = np_utils.to_categorical(flickr_test_set_label, 20)

    if class_type == 'categorical':
        Y_train = np_utils.to_categorical(flickr_train_set_label, 20)
        Y_test = np_utils.to_categorical(flickr_test_set_label, 20)

        return (X_train, Y_train), (X_test, Y_test)
    else:
        y_train = flickr_train_set_label
        y_test = flickr_test_set_label

        return (X_train, y_train), (X_test, y_test)


def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Take
    filename: string
    color: flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Give
    image: an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

if __name__ == '__main__':
    flickr_train_set = np.loadtxt('/shared/flickr_style/train_resized.txt', str, delimiter='\t')
    flickr_train_set = flickr_train_set[:63000]
    flickr_train_set_path = [readline.split()[0] for readline in flickr_train_set]
    flickr_train_set_label = [int(readline.split()[1]) for readline in flickr_train_set]

    flickr_test_set = np.loadtxt('/shared/flickr_style/test_resized.txt', str, delimiter='\t')
    flickr_test_set = flickr_test_set[:15000]
    flickr_test_set_path = [readline.split()[0] for readline in flickr_test_set]
    flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]

    #load_feat_vec()
    #make_batch_pickle(flickr_train_set_path, batch_size=252, rootpath='/shared/flickr_style/train252/')
    make_batch_pickle(flickr_test_set_path, batch_size=60, rootpath='/shared/flickr_style/test60/')
