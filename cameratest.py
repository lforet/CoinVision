import cv

# create windows
cv.NamedWindow('Camera', cv.CV_WINDOW_AUTOSIZE)

# create capture device
device = 0 # assume we want first device
capture = cv.CreateCameraCapture(0)

#for i in range (0,20):
#	print cv.GetCaptureProperty(capture, i)

cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 1600)
cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 1200)  

# capture the current frame
frame = cv.QueryFrame(capture)

# display webcam image
cv.ShowImage('Camera', frame)
cv.SaveImage("testpic.png", frame)

cv.WaitKey()

