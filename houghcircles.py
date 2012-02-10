import sys, os
from opencv import cv
from opencv import highgui
from opencv.cv import *
from opencv.highgui import *
import Image

pos1 = 0
pos2 = 0
pos3 = 0
pos4 = 0
pos5 = 0
pos6 = 0
pos7 = 0

wname = "Find Circles"

def on_trackbar1(position):
	global pos1 
	global pos2
	global pos3
	global pos4
	global pos5
	global pos6
	global pos7
	global img
	global gray
	global edges
	print
	print position, pos2, pos3, pos4, pos5, pos6, pos7

	temp = cv.cvCloneImage(img)
	gray = cv.cvCreateImage(cv.cvGetSize(temp), 8, 1)	
	edges = cv.cvCreateImage(cv.cvGetSize(temp), 8, 1)
	dst =  cv.cvCreateImage( cv.cvSize(256,256), 8, 3 )
	

	src = cv.cvCloneImage(img)
	src2 = cv.cvCreateImage( cv.cvGetSize(src), 8, 3 );
	cv.cvCvtColor(img, gray, cv.CV_BGR2GRAY)

	cv.cvCanny(gray, edges, position, pos2, 3)
	cv.cvSmooth(edges, edges, cv.CV_GAUSSIAN, 9, 9)

	storage = cv.cvCreateMat(50, 1, cv.CV_32FC3)
	cv.cvSetZero(storage)
	try:
		circles = cv.cvHoughCircles(gray, storage, cv.CV_HOUGH_GRADIENT, 1, float(pos3), float(pos2), float(pos4), long(pos5),long(pos6) )
		#print storage
		for i in storage:
			print "Center: ", i[0], i[1], "  Radius: ", i[2]
			center = cv.cvRound(i[0]), cv.cvRound(i[1])
			radius = cv.cvRound(i[2])
			cv.cvCircle(temp, (center), radius, cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 ) 
			cv.cvCircle(edges, (center), radius, cv.CV_RGB(255, 255, 255), 1, cv.CV_AA, 0 ) 
			if radius > 200:
				print "Circle found over 200 Radius"
				center_crop_topleft = (center[0]-(radius - pos7)), (center[1]-(radius - pos7))
				center_crop_bottomright = (center[0]+(radius - pos7)), (center[1]+(radius - pos7))
				print "crop top left:     ", center_crop_topleft
				print "crop bottom right: ", center_crop_bottomright
				center_crop = cv.cvGetSubRect(src, (center_crop_topleft[0], center_crop_topleft[1] , (center_crop_bottomright[0] - center_crop_topleft[0]), (center_crop_bottomright[1] - center_crop_topleft[1])  ))
				#center_crop = cv.cvGetSubRect(src, (50, 50, radius/2, radius/2))
				cvShowImage( "center_crop", center_crop )
				print "center_crop created"
				

				#mark found circle's center with blue point and blue circle of pos 7 radius
				cv.cvCircle(temp ,(center), 2, cv.CV_RGB(0, 0, 255), 3, cv.CV_AA, 0 ) 	
				cv.cvCircle(temp ,(center), (radius - pos7), cv.CV_RGB(0, 0, 255), 3, cv.CV_AA, 0 ) 
				#cvLogPolar(src, dst, (center), 48, CV_INTER_LINEAR	+CV_WARP_FILL_OUTLIERS )
				#this will draw a smaller cirle outlining the center circle				
				#pos7 = int(pos7 /2.5)
				#cv.cvCircle(dst  ,(img_size.width-pos7, 0), 2, cv.CV_RGB(0, 0, 255), 3, cv.CV_AA, 0 )
				#cv.cvLine(dst, (img_size.width-pos7-1, 0), (img_size.width-pos7-1, img_size.height), cv.CV_RGB(0, 0, 255),1,8,0)
				#cvShowImage( "log-polar", dst )
				
				
				#print radius, (radius-pos7)
				
				#cropped = cv.cvCreateImage( (pos7, img_size.height), 8, 3)
				#cropped2 = cv.cvCreateImage( (pos7, img_size.height), 8, 3)
				
				#coin_edge_img = cv.cvGetSubRect(dst, (img_size.width-pos7, 0, pos7 ,img_size.height ))

				#to create the center cropped part of coin
				#img_size = cvGetSize(scr)

				#cvCopy(coin_edge_img, cropped)
				#cvSaveImage("temp.png", cropped)
				#im = Image.open("temp.png").rotate(90)
				#print "pil image size = ", im.size[0], im.size[1]
				#im = im.resize((im.size[0]*2, im.size[1]*2))
				#print "pil image size = ", im.size
				#im.show()
				#im.save("temp2.png")
				cropped2 = highgui.cvLoadImage("temp2.png")
                                #cvShowImage( "cropped", cropped2)

	except:
		print "Exception:", sys.exc_info()[0] 
		print position, pos2, pos3, pos4, pos5, pos6, pos7
		pass

	highgui.cvShowImage("edges", edges)
	#cvShowImage( "log-polar", dst )
	cvShowImage(wname, temp)
	#cvShowImage( "cropped", cropped2)
	

def on_trackbar2(position):
	global pos2 
	global pos1 
	pos2 = position
	pos1 = int(pos2/2)
	highgui.cvSetTrackbarPos("Canny1", wname, pos1)
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

def on_trackbar7(position):
	global pos7
	pos7 = position
	on_trackbar1(pos1)



def on_mouse( event, x, y, flags, param ):

    if( not src ):
        return;

    if event==CV_EVENT_LBUTTONDOWN:
        cvLogPolar( src, dst, cvPoint2D32f(x,y), 40, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS );
        cvLogPolar( dst, src2, cvPoint2D32f(x,y), 40, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP );
        #cvShowImage( "log-polar", dst );
        #cvShowImage( "inverse log-polar", src2 );


if __name__ == "__main__":
	highgui.cvNamedWindow(wname, 1)

	img = highgui.cvLoadImage(sys.argv[1])
	#highgui.cvShowImage("Image", img)

	highgui.cvCreateTrackbar("Canny1", wname, 87, 250, on_trackbar1)
	highgui.cvCreateTrackbar("Canny2", wname, 175, 250, on_trackbar2)
	highgui.cvCreateTrackbar("minDistance", wname, 40, 150, on_trackbar3)
	highgui.cvCreateTrackbar("accumThresh", wname, 55, 100, on_trackbar4)
	highgui.cvCreateTrackbar("minRadius", wname, 190, 500, on_trackbar5)
	highgui.cvCreateTrackbar("maxRadius", wname, 210, 1200, on_trackbar6)
	highgui.cvCreateTrackbar("SearchRadius", wname, 50, 100, on_trackbar7)
	pos1 = highgui.cvGetTrackbarPos("Canny1", wname)
	pos2 = highgui.cvGetTrackbarPos("Canny2", wname)
	pos3 = highgui.cvGetTrackbarPos("minDistance", wname)
	pos4 = highgui.cvGetTrackbarPos("accumThresh", wname)
	pos5 = highgui.cvGetTrackbarPos("minRadius", wname)
	pos6 = highgui.cvGetTrackbarPos("maxRadius", wname)
	pos7 = highgui.cvGetTrackbarPos("SearchRadius", wname)

	pos1 = int(pos2/2)
	highgui.cvSetTrackbarPos("Canny1", wname, pos1)
	on_trackbar1(pos1)

	#highgui.cvNamedWindow( "original",1 );
	#highgui.cvNamedWindow( "log-polar", 1 );
	#highgui.cvNamedWindow( "inverse log-polar", 1 );

	dst = cv.cvCreateImage( cv.cvSize(256,256), 8, 3 );
	src = cv.cvCloneImage(img)
	src2 = cv.cvCreateImage( cv.cvGetSize(src), 8, 3 );
        #cvShowImage( "original", src );
	highgui.cvSetMouseCallback( "original", on_mouse );
	on_mouse( CV_EVENT_LBUTTONDOWN, src.width/2, src.height/2, None, None)

	highgui.cvWaitKey()
