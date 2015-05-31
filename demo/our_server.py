import cherrypy
import numpy as np
import caffe
import json
import recommender as recom
from bonggunet import BongguNet

caffe_root = '/home/hawk/imagenet2014/caffe/'
flickr_test_set = np.loadtxt(caffe_root + 'data/flickr_style/test.txt', str, delimiter='\t')
flickr_test_set_path = [readline.split()[0] for readline in flickr_test_set]
flickr_test_set_label = [int(readline.split()[1]) for readline in flickr_test_set]
flickr_train_set = np.loadtxt(caffe_root + 'data/flickr_style/train.txt', str, delimiter='\t')
flickr_train_set_path = [readline.split()[0] for readline in flickr_test_set]
flickr_train_set_label = [int(readline.split()[1]) for readline in flickr_test_set]


class FlickrServer(object):
    def __init__(self):
        self.our_model = BongguNet()
        print "model initialized"

    @cherrypy.expose
    def index(self, path=flickr_test_set_path[9]):
        print "index page, sample path with", path
        img = caffe.io.load_image(path)
        print "start predict"
        res = self.our_model.predict(img)
        res["recommended"] = recom.recommender(res)
        print res
        print "end predict"
        return json.dumps(res)


if __name__ == '__main__':
    #test_model()
    cherrypy.config.update({'server.socket_port': 8080,
                            'server.socket_host': '0.0.0.0',
                            'engine.autoreload_on': False,
                            'log.access_file': './access.log',
                            'log.error_file': './error.log'})
    cherrypy.quickstart(FlickrServer())
