#!/usr/bin/python


"""
public static IntPtr cvHoughCircles(
	IntPtr image,
	IntPtr circleStorage,
	HOUGH_TYPE method,
	double dp,
	double minDist,
	double param1,
	double param2,
	int minRadius,
	int maxRadius
)

Public Shared Function cvHoughCircles ( _
	image As IntPtr, _
	circleStorage As IntPtr, _
	method As HOUGH_TYPE, _
	dp As Double, _
	minDist As Double, _
	param1 As Double, _
	param2 As Double, _
	minRadius As Integer, _
	maxRadius As Integer _
) As IntPtr


Parameters

image (IntPtr)
    The input 8-bit single-channel grayscale image

circleStorage (IntPtr)
    The storage for the circles detected. It can be a memory storage (in this case a sequence of circles is created in the storage and returned by the function) or single row/single column matrix (CvMat*) of type CV_32FC3, to which the circles' parameters are written. The matrix header is modified by the function so its cols or rows will contain a number of lines detected. If circle_storage is a matrix and the actual number of lines exceeds the matrix size, the maximum possible number of circles is returned. Every circle is encoded as 3 floating-point numbers: center coordinates (x,y) and the radius

method (HOUGH_TYPE)
    Currently, the only implemented method is CV_HOUGH_GRADIENT

dp (Double)
    Resolution of the accumulator used to detect centers of the circles. For example, if it is 1, the accumulator will have the same resolution as the input image, if it is 2 - accumulator will have twice smaller width and height, etc

minDist (Double)
    Minimum distance between centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed

param1 (Double)
    The first method-specific parameter. In case of CV_HOUGH_GRADIENT it is the higher threshold of the two passed to Canny edge detector (the lower one will be twice smaller). 

param2 (Double)
    The second method-specific parameter. In case of CV_HOUGH_GRADIENT it is accumulator threshold at the center detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first

minRadius (Int32)
    Minimal radius of the circles to search for

maxRadius (Int32)
    Maximal radius of the circles to search for. By default the maximal radius is set to max(image_width, image_height). 
"""

import sys, os
from opencv import cv
from opencv import highgui

print "to use: python houghcircles.py imagefile.jpg minRadius maxRadius"

# first, create the necessary window
highgui.cvStartWindowThread()
highgui.cvNamedWindow('Image Display Window', highgui.CV_WINDOW_AUTOSIZE)
highgui.cvNamedWindow('GrayScale', highgui.CV_WINDOW_AUTOSIZE)
highgui.cvNamedWindow('Canny', highgui.CV_WINDOW_AUTOSIZE)

# move the new window to a better place
highgui.cvMoveWindow ('Image Display Window', 10, 10)
highgui.cvMoveWindow ('GrayScale', 100, 10)
highgui.cvMoveWindow ('Canny', 200, 10)

#load image
image = highgui.cvLoadImage(sys.argv[1]);

#create image arrays
grayimage = cv.cvCreateImage(cv.cvGetSize(image), 8, 1)
cannyedges = cv.cvCreateImage(cv.cvGetSize(image), 8, 1)


#convert to grayscale
cv.cvCvtColor(image, grayimage, cv.CV_BGR2GRAY)
#Canny
#Canny(image, edges, threshold1, threshold2, aperture_size=3) = None
#Implements the Canny algorithm for edge detection.
cv.cvCanny(grayimage, cannyedges, 150, 450 , 3)


#This is the line that throws the error
storage = cv.cvCreateMat(50, 1, cv.CV_32FC3)

cv.cvSetZero(storage)

#circles = cv.cvHoughCircles(grayimage, storage, cv.CV_HOUGH_GRADIENT, 2, grayimage.height/4, 150, 40, long(sys.argv[2]), long(sys.argv[3]))
circles = cv.cvHoughCircles(grayimage, storage, cv.CV_HOUGH_GRADIENT, 2, 500, 200, 40, long(sys.argv[2]), long(sys.argv[3]))

print storage
for i in storage:
    print i[0], i[1], i[2]
    center = cv.cvRound(i[0]), cv.cvRound(i[1])
    radius = cv.cvRound(i[2])
    cv.cvCircle(image, (center), radius, cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 ) 
    cv.cvCircle(cannyedges, (center), radius, cv.CV_RGB(255, 255, 255), 1, cv.CV_AA, 0 ) 
 #   for c in range(0,3):
        #v = cv.cvRound( cv.cvmGet(storage, i, c) )
        #if v != 0.0:
            #print v


#cv.cvSmooth(grayimage, grayimage,cv.CV_MEDIAN, 3, 3)

#Display image
highgui.cvShowImage ('Image Display Window', image)
highgui.cvShowImage ('GrayScale', grayimage)
highgui.cvShowImage ('Canny', cannyedges)

#wait for key press then exit
while 1:
    # handle events
    k = highgui.cvWaitKey()
    break
sys.exit()
