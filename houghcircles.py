import cv
import sys

pos2 = 0
pos1 = 0
wname = "Window"

def on_trackbar1(position):
	global pos2
	print position, pos2
	img = cv.LoadImage(sys.argv[1])
	gray = cv.CreateImage(cv.GetSize(img), 8, 1)
	edges = cv.CreateImage(cv.GetSize(img), 8, 1)
	cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

	cv.Canny(gray, edges, 50, 150, 3)
	cv.Smooth(gray, gray, cv.CV_GAUSSIAN, 9, 9)
	storage = cv.CreateMat(1, 2, cv.CV_32FC3)
	cv.ShowImage(wname, gray)
	cv.ShowImage("edge", edges)

	print cv.HoughCircles(edges, storage, cv.CV_HOUGH_GRADIENT, 2, edges.height, position, pos2) 
	

def on_trackbar2(position):
	global pos2 
	pos2 = position
	print pos2
	on_trackbar1(0)



cv.NamedWindow(wname, 1)

cv.CreateTrackbar("trackbar1", wname, 0, 250, on_trackbar1)
cv.CreateTrackbar("trackbar2", wname, 0, 250, on_trackbar2)
cv.WaitKey()
