#!/usr/bin/env python

#This program will return the angle at which the second is in relation to the first. 
#params: arg1 = base or original image, arg2= image that is mis-oriented


import cv
#from opencv import cv2
import sys
import numpy as np
import Image 
import math, operator
import time
import scipy.spatial
import ImageChops
import ImageOps
from math import pi

def find_center_of_coin(img):
	#create storage fo circle data
	storage = cv.CreateMat(50, 1, cv.CV_32FC3)
 	#storage = cv.CreateMemStorage(0)

	cv.SetZero(storage)
	
	edges = cv.CreateImage(cv.GetSize(img), 8, 1)
	cv.Smooth(img , edges , cv.CV_GAUSSIAN,3, 3)
	cv.Canny(edges, edges, 87, 187, 3)
    	#cv.Smooth(edges, edges, cv.CV_GAUSSIAN, 3, 3)

	cv.ShowImage("grayed center image", edges)
	cv.WaitKey()

	circles = cv.HoughCircles(edges, storage, cv.CV_HOUGH_GRADIENT, 1, img.width, 187, 30, 150 , img.width)

	for i in range(0,len(np.asarray(storage))):
    		center = int(np.asarray(storage)[i][0][0]), int(np.asarray(storage)[i][0][1])
		radius = int(np.asarray(storage)[i][0][2])
		print  center, radius
    		cv.Circle(img, (center), radius, cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 ) 
    		cv.Circle(img, (center), 10, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 ) 
    		#cv.Circle(cannyedges, (center), radius, cv.CV_RGB(255, 255, 255), 1, cv.CV_AA, 0 )
	cv.ShowImage("Center of Coin", img)
	#need code here to validate center found: like has to be within x,y of center of image
	cv.WaitKey()


def get_orientation(img1, img2): 

	subtracted_image = cv.CreateImage(cv.GetSize(img1), 8, 1)
	temp_img = cv.CreateImage(cv.GetSize(img1), 8, 1)

	best_sum = 0
	best_orientation = 0
	print 'Starting to find best orientation'
	for i in range(1, 360):
		temp_img = rotate_image(img2, i)
		cv.And(img1, temp_img , subtracted_image)
		cv.ShowImage("Image of Interest", temp_img )
		cv.ShowImage("subtracted_image", subtracted_image)
		sum_of_and = cv.Sum(subtracted_image)
		if best_sum == 0: best_sum = sum_of_and[0]
		if sum_of_and[0] > best_sum: 
			best_sum = sum_of_and[0]
			best_orientation = i
		print i, "Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break
		time.sleep(.05)
	print 'Finished finding best orientation'
	return (best_orientation)

def rotate_image(img, degrees):
	"""
    rotate(scr1, degrees) -> image
    Parameters:	

         *  image - source image
         *  angle (integer) - The rotation angle in degrees. Positive values mean counter-clockwise 	rotation 
	"""
	temp_img = cv.CreateImage(cv.GetSize(img), 8, img.channels)
	mapMatrix = cv.CreateMat( 2, 3, cv.CV_32FC1 )
	img_size = cv.GetSize(img)
	img_center = (int(img_size[0]/2), int(img_size[1]/2))
	cv.GetRotationMatrix2D(img_center, degrees, 1.0, mapMatrix)
	cv.WarpAffine(img , temp_img, mapMatrix, flags=cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))
	return(temp_img)



if __name__=="__main__":

	if len(sys.argv) < 3:
		print "******* Requires 2 image files of the same size."
		print "This program will return the angle at which the second is in relation to the first. ***"
		sys.exit(-1)

	try:
		img1 = cv.LoadImage(sys.argv[1],cv.CV_LOAD_IMAGE_GRAYSCALE)
		img2 = cv.LoadImage(sys.argv[2],cv.CV_LOAD_IMAGE_GRAYSCALE)
	except:
		print "******* Could not open image files *******"
		sys.exit(-1)

	img1_size  = cv.GetSize(img1)
	img1_width = img1_size[0]
	img1_height = img1_size[1]
	img2_size  = cv.GetSize(img2)
	img2_width = img2_size[0]
	img2_height = img2_size[1]

	if img1_size <> img2_size:
		print "Images must be of the same size........End Of Line/"
		sys.exit(-1)


	cv.ShowImage("Image 1", img1)
	cv.ShowImage("Image 2", img2)
	cv.WaitKey()

	find_center_of_coin(img1)	
	sys.exit(-1)

	img1_copy = cv.CloneImage(img1)
	img2_copy = cv.CloneImage(img2)
	cv.Smooth(img1_copy , img1_copy , cv.CV_GAUSSIAN,3, 3)
	cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN, 3, 3)
	cv.Canny(img1_copy ,img1_copy  ,87,175, 3)
	cv.Canny(img2_copy, img2_copy , 87,175, 3)

	cv.ShowImage("img1_copy ", img1_copy )
	cv.ShowImage("img2_copy ", img2_copy )
	cv.WaitKey()

	best_orientation = (0,0)
	degrees = get_orientation(img1_copy, img2_copy)

	img3 = cv.CloneImage(img2_copy)
	img3 = rotate_image(img2_copy, degrees)
	cv.ShowImage("Corrected Image2", img3 )
	cv.WaitKey()
