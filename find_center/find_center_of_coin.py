from SimpleCV import *
import time
import webbrowser

c = Camera()
d = Display()

js = JpegStreamer(("192.168.1.88", 8080))  #starts up an http server (defaults to port 8080)
#js = JpegTCPServer(("localhost", 8080), JpegStreamHandler)


#webbrowser.open(js.url())
#print js.url()
#print js.streamUrl()

while (1): 
	c.getImage().save(js)
	c.getImage().save(d)
	#d.show()
	time.sleep(0.1)


