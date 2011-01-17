#!/usr/bin/env python

#This program will find and return the center of a coin image. It assumes only one coin per image

import cv
import optparse


# definition of some colors
_red =  (0, 0, 255, 0);
_green =  (0, 255, 0, 0);



#------Add option parameters	 
parser = optparse.OptionParser()

parser.add_option('-f', help='filename', dest='image',
    action='store_true')
#parser.add_option('-p', help='mandatory option', dest='pan',
#    action='store_true')

(opts, args) = parser.parse_args()

# Making sure all mandatory options appeared.
#mandatories = ['image', 'pan']
mandatories = ['image']
for m in mandatories:
    if not opts.__dict__[m]:
        print "mandatory option is missing\n"
        parser.print_help()
        exit(-1)

#print opts.image, args[0]


#load image
img = cv.LoadImageM(args[0], cv.CV_LOAD_IMAGE_GRAYSCALE)

# create the canny image
canny_image = cv.CreateImage (cv.GetSize (img), 8, 1)
dilated_image = cv.CreateImage (cv.GetSize (img), 8, 1)
binary_image = cv.CreateImage (cv.GetSize (img), 8, 1)
smooth_image = cv.CreateImage (cv.GetSize (img), 8, 1)
final_image = cv.CreateImage (cv.GetSize (img), 8, 1)
temp_image = cv.CreateImage (cv.GetSize (img), 8, 1)

#cv.Threshold(img, binary_image, 128, 255, cv.CV_THRESH_BINARY)
#cv.Canny(img, canny_image , 50 , 150)
#cv.ShowImage("Canny",canny_image )
#cv.Smooth(img, smooth_image, smoothtype=cv.CV_MEDIAN , param1=3, param2=0, param3=0, param4=0)
#cv.Erode(img, smooth_image, iterations=5)
#cv.Dilate(canny_image, dilated_image, iterations=5)
cv.Smooth(img, final_image, smoothtype=cv.CV_MEDIAN , param1=3, param2=0, param3=0, param4=0)
cv.Threshold(final_image, final_image, 128, 255, cv.CV_THRESH_BINARY)
#cv.MorphologyEx(final_image, final_image, temp_image, element, operation, iterations=1)
cv.Erode(final_image, final_image, iterations=2)
cv.Dilate(final_image, final_image, iterations=2)



cv.ShowImage("Image",img )
#cv.ShowImage("Smoothed",smooth_image )
#cv.ShowImage("Dilated",dilated_image )
#cv.ShowImage("Binary",binary_image )


# create the storage area
storage = cv.CreateMemStorage (0)

# find the contours
contour = cv.FindContours(final_image, storage,cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE, (0,0))

# comment this out if you do not want approximation
contour = cv.ApproxPoly (contour, storage, cv.CV_POLY_APPROX_DP, 3, 1)

# draw contours in red and green
cv.DrawContours (img, contour, _red, _green,8 ,2, cv.CV_AA,(0, 0))


#For each contour found
points = []


while contour:
	bound_rect = cv.BoundingRect(list(contour))
	#print bound_rect,  list(contour)

	# Get the size of the contour
	size = abs(cv.ContourArea(contour))
	
	# Is convex
	is_convex = cv.CheckContourConvexity(contour)
	print "size = ", size , " is convex = ", is_convex
	# Find the bounding-box of the contour
	bbox = cv.BoundingRect(list(contour))

	# Calculate the x and y coordinate of center
	##x, y = (bbox.x + bbox.width * 0.5), (bbox.y+bbox.height*0.5)
	#print (bbox[0] + bbox[1] *0.5), 

	#print "is_convex = ", is_convex, "   Center = ", x ,y 



	contour = contour.h_next()
 
	pt1 = (bound_rect[0], bound_rect[1])
	pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
	points.append(pt1)
	points.append(pt2) 
	cv.Rectangle(img, pt1, pt2, cv.CV_RGB(255,0,0), 3)



#if len(points):
#                center_point = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), points)
#                cv.Circle(img, center_point, 40, cv.CV_RGB(255, 255, 255), 1)
#                cv.Circle(img, center_point, 30, cv.CV_RGB(255, 100, 0), 1)
#                cv.Circle(img, center_point, 20, cv.CV_RGB(255, 255, 255), 1)
#                cv.Circle(img, center_point, 10, cv.CV_RGB(255, 100, 0), 1)



cv.ShowImage("Image",img )

cv.WaitKey(0)

def Despeckle(image):
    """
    Reduce speckle noise. This filter works quite nice with minimal smoothing 
    of edges.
    """
    img = PythonMagick.Image(image) # copy
    img.despeckle()
    return img   
