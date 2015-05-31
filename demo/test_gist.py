import numpy as np
import leargist
import caffe

caffe_root = '../caffe/'
project_root = '../'
style_dic = np.loadtxt(caffe_root + 'examples/finetune_flickr_style/style_names.txt', str, delimiter='\t')
synset_dic = np.loadtxt(caffe_root + 'data/flickr_style/synset_words.txt', str, delimiter='\t')

def test_gist():
    flickr_test_set = np.loadtxt(caffe_root + 'data/flickr_style/test.txt', str, delimiter='\t')
    flickr_test_set_path = [readline.split()[0] for readline in flickr_test_set]
    flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]

    n = 9
    img = caffe.io.load_image(flickr_test_set_path[n])
    print "true label", flickr_test_set_label[n]

    print leargist.color_gist(img)
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    #gist_features[0, :] = leargist.color_gist(img)

    #print gist_features.shape

if __name__ == '__main__':
    test_gist()
