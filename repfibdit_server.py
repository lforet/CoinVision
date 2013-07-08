import SimpleHTTPServer
import SocketServer

theport = 1234
Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
pywebserver = SocketServer.TCPServer(("", theport), Handler)

print "Python based web server. Serving at port", theport
pywebserver.serve_forever()
