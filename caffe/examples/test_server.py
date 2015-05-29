import cherrypy

class Root(object):
    @cherrypy.expose
    def index(self):
        return "Hello World!"

cherrypy.config.update({'server.socket_port': 8090,
                        'server.socket_host': '0.0.0.0',
                        'engine.autoreload_on': False,
                        'log.access_file': './access.log',
                        'log.error_file': './error.log'})
cherrypy.quickstart(Root())
