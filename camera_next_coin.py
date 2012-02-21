import cv
import time
import math


cv.NamedWindow('Camera', cv.CV_WINDOW_AUTOSIZE)
capture =  cv.CreateCameraCapture(0)
time.sleep(.05)

cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 320)
cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 240)  

while 1:
	
	frame = cv.QueryFrame(capture)
	cv.ShowImage('Camera', frame)
	result1 = cv.Sum(frame)
	key = cv.WaitKey(5)
	
	if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
		break 
	#time.sleep(.05)
	frame2 = cv.QueryFrame(capture)
	cv.ShowImage('Camera', frame2)
	result2 = cv.Sum(frame2)
	print "mean:", (result2[0] / (frame.height * frame.width))
	#dif = math.fabs(result2[0] - result1[0])
	dif = (result2[0]/ result1[0])
	print dif
	if dif < .90 or dif > 1.10:
		print "new coin", 
	key = cv.WaitKey(5)
	if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
		break 
	#time.sleep(.05)







