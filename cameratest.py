import cv2
import cv
import time

def get_camera_prop_name(prop_num):
	prop_str = ""

	#Property identifier. It can be one of the following:
	if prop_num == 0: prop_str = "CV_CAP_PROP_POS_MSEC" #Current position of the video file in milliseconds.
	if prop_num == 1: prop_str = "CV_CAP_PROP_POS_FRAMES" # 0-based index of the frame to be decoded/captured next.
	if prop_num == 2: prop_str = "CV_CAP_PROP_POS_AVI_RATIO" # Relative position of the video file: 0 - start of the film, 1 - end of the film.
	if prop_num == 3: prop_str = "CV_CAP_PROP_FRAME_WIDTH" # Width of the frames in the video stream.
	if prop_num == 4: prop_str = "CV_CAP_PROP_FRAME_HEIGHT" # Height of the frames in the video stream.
	if prop_num == 5: prop_str = "CV_CAP_PROP_FPS" # Frame rate.
	if prop_num == 6: prop_str = "CV_CAP_PROP_FOURCC" # 4-character code of codec.
	if prop_num == 7: prop_str = "CV_CAP_PROP_FRAME_COUNT" # Number of frames in the video file.
	if prop_num == 8: prop_str = "CV_CAP_PROP_FORMAT" # Format of the Mat objects returned by retrieve() .
	if prop_num == 9: prop_str = "CV_CAP_PROP_MODE" # Backend-specific value indicating the current capture mode.
	if prop_num == 10: prop_str = "CV_CAP_PROP_BRIGHTNESS" # Brightness of the image (only for cameras).
	if prop_num == 11: prop_str = "CV_CAP_PROP_CONTRAST" # Contrast of the image (only for cameras).
	if prop_num == 12: prop_str = "CV_CAP_PROP_SATURATION" # Saturation of the image (only for cameras).
	if prop_num == 13: prop_str = "CV_CAP_PROP_HUE" # Hue of the image (only for cameras).
	if prop_num == 14: prop_str = "CV_CAP_PROP_GAIN" # Gain of the image (only for cameras).
	if prop_num == 15: prop_str = "CV_CAP_PROP_EXPOSURE" # Exposure (only for cameras).
	if prop_num == 16: prop_str = "CV_CAP_PROP_CONVERT_RGB" # Boolean flags indicating whether images should be converted to RGB.
	if prop_num == 17: prop_str = "CV_CAP_PROP_WHITE_BALANCE" # Currently unsupported
	if prop_num == 18: prop_str = "CV_CAP_PROP_RECTIFICATION" # Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
	return prop_str



# create windows
cv2.namedWindow('Bottom Camera', cv.CV_WINDOW_AUTOSIZE)
cv2.namedWindow('Top Camera', cv.CV_WINDOW_AUTOSIZE)
#cv2.waitKey()
# create capture device
Top_Camera= 1 # assume we want first device
Bottom_Camera = 0
#device = 1 # assume we want first device
#top_capture = cv.CreateCameraCapture(Top_Camera)
#bottom_capture = cv.CreateCameraCapture(Bottom_Camera)
#capture = cv2.VideoCapture(device)


#print capture
#print capture.isOpened()
#cv2.waitKey()

#for i in range (0,20):
	#print cv.GetCaptureProperty(capture, i)
#	print i, "= ", capture.get(i)

#cv.WaitKey()
#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 1600)
#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 1200)  

#print capture.set(3, 320)
#print capture.set(4, 240)

#for i in range (0,20):
	#print cv.GetCaptureProperty(capture, i)
#	print get_camera_prop_name(i), "= ", capture.get(i)
# capture the current frame
#frame = cv.QueryFrame(capture)
#frame = capture.read()
#frame2 = cv.ImageClone(frame)
#display webcam image
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


#cv.SaveImage("images/tail_1.jpg", frame)
#cv.SaveImage("images/head_1.jpg", frame2)
#cv2.imshow("w", frame)


