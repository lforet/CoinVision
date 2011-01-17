import sys, os
from opencv import cv
from opencv import highgui

pos1 = 0
pos2 = 0
pos3 = 0
pos4 = 0
pos5 = 0
pos6 = 0
wname = "Window"

def on_trackbar1(position):
	global pos1 
	pos1 = position
	global pos2
	global pos3
	global pos4
	global pos5
	global pos6
	global img
	global gray
	global edges
	#print position, pos2, pos3, pos4, pos5, pos6
	gray = cv.cvCreateImage(cv.cvGetSize(img), 8, 1)
	edges = cv.cvCreateImage(cv.cvGetSize(img), 8, 1)
	cv.cvCvtColor(img, gray, cv.CV_BGR2GRAY)

	cv.cvCanny(gray, edges, position, pos2, 3)
	cv.cvSmooth(edges, edges, cv.CV_GAUSSIAN, 9, 9)

	storage = cv.cvCreateMat(50, 1, cv.CV_32FC3)
	cv.cvSetZero(storage)
	try:
		circles = cv.cvHoughCircles(gray, storage, cv.CV_HOUGH_GRADIENT, 1, float(pos3), float(pos2), float(pos4), long(pos5),long(pos6) )
		#print storage
		for i in storage:
		    print i[0], i[1], i[2]
		    center = cv.cvRound(i[0]), cv.cvRound(i[1])
		    radius = cv.cvRound(i[2])
		    cv.cvCircle(gray, (center), radius, cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 ) 
		    cv.cvCircle(edges, (center), radius, cv.CV_RGB(255, 255, 255), 1, cv.CV_AA, 0 ) 
	except:
		print sys.exc_info()[0]
		print position, pos2, pos3, pos4, pos5, pos6
		pass

	highgui.cvShowImage("gray", gray)
	highgui.cvShowImage("edges", edges)

def on_trackbar2(position):
	global pos2 
	pos2 = position
	on_trackbar1(pos1)

def on_trackbar3(position):
	global pos3
	pos3 = position
	on_trackbar1(pos1)

def on_trackbar4(position):
	global pos4
	pos4 = position
	on_trackbar1(pos1)
	
def on_trackbar5(position):
	global pos5
	pos5 = position
	on_trackbar1(pos1)

def on_trackbar6(position):
	global pos6
	pos6 = position
	on_trackbar1(pos1)

highgui.cvNamedWindow(wname, 1)

img = highgui.cvLoadImage(sys.argv[1])
highgui.cvShowImage("Image", img)

highgui.cvCreateTrackbar("Canny1", wname, 50, 250, on_trackbar1)
highgui.cvCreateTrackbar("Canny2", wname, 175, 250, on_trackbar2)
highgui.cvCreateTrackbar("minDistance", wname, 40, 150, on_trackbar3)
highgui.cvCreateTrackbar("accumThresh", wname, 55, 100, on_trackbar4)
highgui.cvCreateTrackbar("minRadius", wname, 190, 500, on_trackbar5)
highgui.cvCreateTrackbar("maxRadius", wname, 210, 1200, on_trackbar6)
pos1 = highgui.cvGetTrackbarPos("Canny1", wname)
pos2 = highgui.cvGetTrackbarPos("Canny2", wname)
pos3 = highgui.cvGetTrackbarPos("minDistance", wname)
pos4 = highgui.cvGetTrackbarPos("accumThresh", wname)
pos5 = highgui.cvGetTrackbarPos("minRadius", wname)
pos6 = highgui.cvGetTrackbarPos("maxRadius", wname)

on_trackbar1(50)

highgui.cvWaitKey()
