import cv
import time


cv.namedWindow('Camera', cv.CV_WINDOW_AUTOSIZE)

while 1:
	capture =  cv.CreateCameraCapture(Top_Camera)
	frame = cv.QueryFrame(capture)
	cv.ShowImage('Top Camera', frame)
	cv.SaveImage("images/tail_1.jpg", frame)
	key = cv.WaitKey(50)
	time.sleep(1)
	capture = cv.CreateCameraCapture(Bottom_Camera)
	frame = cv.QueryFrame(capture)
	cv.ShowImage('Bottom Camera', frame)
	cv.SaveImage("images/head_1.jpg", frame)
	#time.sleep(.05)
	key = cv.WaitKey(50)
	time.sleep(1)
	if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
		break 

