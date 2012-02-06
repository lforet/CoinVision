from SimpleCV import *
import time

img = Image("../images/left.jpg")

disp = Display()

while disp.isNotDone():
        if disp.mouseLeft:
	'''
	Smooth the image, by default with the Gaussian blur. If desired, additional algorithms and aperatures can be specified. Optional parameters are passed directly to OpenCV’s cv.Smooth() function.
If grayscale is true the smoothing operation is only performed on a single channel otherwise the operation is performed on each channel of the image.
Returns: IMAGE
	'''
		img = img.smooth()
                img = img.edges(187,87)
		time.sleep(1)
        if disp.mouseRight:
                break
        img.save(disp)
