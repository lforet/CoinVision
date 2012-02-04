#!/usr/bin/env python

#This program will return the angle at which the second is in relation to the first. 
#params: arg1 = base or original image, arg2= image that is mis-oriented


import cv
#from opencv import cv2
from SimpleCV import *
import sys
import numpy as np
import Image 
import math, operator
import time
import scipy.spatial
import ImageChops
import ImageOps
from math import pi



#def resize_img(original_img, scale_percentage):
 

#	resized_img = cv.CreateMat(original.rows / 10, original.cols / 10, cv.CV_8UC3)
#	cv.Resize(original, thumbnail)



def center_crop(img, center, crop_size):
	#crop out center of coin based on found center
	x,y = center[0][0], center[0][1]
	#radius = center[1]
	radius = 200
	center_crop_topleft = (x-(radius-crop_size), y-(radius-crop_size))
	center_crop_bottomright = (x+(radius-crop_size), y+(radius-crop_size))
	print "crop top left:     ", center_crop_topleft
	print "crop bottom right: ", center_crop_bottomright
	center_crop = cv.GetSubRect(img, (center_crop_topleft[0], center_crop_topleft[1] , (center_crop_bottomright[0] - center_crop_topleft[0]), (center_crop_bottomright[1] - center_crop_topleft[1])  ))
	#cv.ShowImage("Crop Center of Coin", center_crop)
	#cv.WaitKey()
	return center_crop


def find_center_of_coin(img):
	#create storage fo circle data
	storage = cv.CreateMat(50, 1, cv.CV_32FC3)
 	#storage = cv.CreateMemStorage(0)

	cv.SetZero(storage)
	
	edges = cv.CreateImage(cv.GetSize(img), 8, 1)
	print edges, img
	cv.Smooth(img , edges , cv.CV_GAUSSIAN,3, 3)
	#cv.Canny(edges, edges, 50, 100, 3)
    	#cv.Smooth(edges, edges, cv.CV_GAUSSIAN, 3, 3)

	#cv.ShowImage("grayed center image", edges)
	#cv.WaitKey()

	best_circle = ((0,0),0)
	for minRadius in range (160, 220, 2):
		for maxRadius in range (236, 250, 4):
			print "minRadius: ", minRadius, "  maxRadius: ", maxRadius
			circles = cv.HoughCircles(edges, storage, cv.CV_HOUGH_GRADIENT, 1, img.width, 100, 30, minRadius, maxRadius)

			if len(np.asarray(storage)) > 0:
				for i in range(0,len(np.asarray(storage))):
	    				center = int(np.asarray(storage)[i][0][0]), int(np.asarray(storage)[i][0][1])
					radius = int(np.asarray(storage)[i][0][2])
					#print  center, radius	
					if (radius > best_circle[1]) & (radius < 216) :
						best_circle = (center, radius)
						print "Found Best Circle: ", best_circle, "   minRadius: ", minRadius, "  maxRadius: ", maxRadius
	

    						cv.Circle(img, (best_circle[0]), best_circle[1], cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 ) 
					    	cv.Circle(img, (best_circle[0]), 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 ) 
					    	#cv.Circle(cannyedges, (center), radius, cv.CV_RGB(255, 255, 255), 1, cv.CV_AA, 0 )
						cv.ShowImage("Center of Coin", img)
						#need code here to validate center found: like has to be within x,y of center of image
						cv.WaitKey(10)
						#time.sleep(.1)
						#cv.WaitKey()
	return best_circle

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

	img1_copy = cv.CloneImage(img1)
	img2_copy = cv.CloneImage(img2)

	#find center of img1 coin
	print "Getting center of both coins"
	coin1_center = find_center_of_coin(img1_copy)
	#print coin1_center	
	#print coin1_center[0][0], coin1_center[0][1], coin1_center[1]
	#
	#cv.WaitKey()
	coin2_center = find_center_of_coin(img2_copy)	
	if coin2_center[1] < coin1_center[1]:
		print coin1_center[1], coin2_center[1]
		scale = float(coin1_center[1]) / float(coin2_center[1])
		print "must scale img2 up%:", scale
		temp_img = SimpleCV.Image(sys.argv[2]).toGray()
		
		scaled_img = temp_img.scale(scale)
		#scaled_img = scaled_img.grayscale()
		scaled_img = scaled_img.getBitmap()
		cv.ShowImage("scaled", scaled_img)
		temp_gray = cv.CreateImage(cv.GetSize(scaled_img), 8, 1)
		cv.CvtColor(scaled_img, temp_gray, cv.CV_RGB2GRAY)
		temp_gray_copy = cv.CloneImage(temp_gray)
		coin2_center = find_center_of_coin(temp_gray_copy)
	#cv.WaitKey()
	#sys.exit(-1)


	#crop out center of coin based on found center
	coin1_center_crop = center_crop(img1, coin1_center, 50)
	cv.ShowImage("Crop Center of Coin1", coin1_center_crop)
	cv.MoveWindow ('Crop Center of Coin1', 100, 100)
	#cv.WaitKey()
	coin2_center_crop = center_crop(temp_gray, coin2_center, 50)
	cv.ShowImage("Crop Center of Coin2", coin2_center_crop)
	cv.MoveWindow ('Crop Center of Coin2', 100, (120 + (cv.GetSize(coin1_center_crop)[0])) )
	cv.WaitKey()

	#obj_size = cv.GetSize(coin1_center_crop) 
	#obj_width = obj_size[0]
	#obj_height = obj_size[1]
	#img1_copy = cv.CreateImage( (cv.GetSize(coin1_center_crop)),img2_copy1)
	#img2_copy = cv.CreateImage( (cv.GetSize(coin2_center_crop)), 8, 1)
	#temp_img  = cv.CreateImage( (cv.GetSize(coin2_center_crop)), 8, 1)
	#print img1_copy, img2_copy
 	
	#img1_copy = cv.GetImage(coin1_center_crop)
	#img2_copy = cv.GetImage(coin2_center_crop)
	#cv.CvtColor(img2_copy, temp_img, cv.CV_RGB2GRAY)

	img1_copy = cv.CloneMat(coin1_center_crop)
	img2_copy = cv.CloneMat(coin2_center_crop)
	
	print "mats?:", coin1_center_crop, coin2_center_crop
	print "these are the cropped images: ", img1_copy, img2_copy
	cv.WaitKey()

	cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
	cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN, 3, 3)
	cv.Canny(img1_copy ,img1_copy  ,87,187, 3)
	cv.Canny(img2_copy, img2_copy , 87,187, 3)

	cv.ShowImage  ("Canny Coin 1", img1_copy )
	cv.MoveWindow ('Canny Coin 1', (100 + (1 * (cv.GetSize(coin1_center_crop)[0]))) , 100)
	cv.ShowImage  ("Canny Coin 2", img2_copy )
	cv.MoveWindow ('Canny Coin 2', (100 + (1 * (cv.GetSize(coin1_center_crop)[0]))) , (100 + (cv.GetSize(coin1_center_crop)[0])) )
	cv.WaitKey()

	best_orientation = (0,0)
	degrees = get_orientation(img1_copy, img2_copy)
	print "Degrees Re-oriented: ", degrees
	img3 = cv.CloneMat(coin2_center_crop)
	img3 = rotate_image(coin2_center_crop, degrees)
	cv.ShowImage("Corrected Image2", img3 )
	cv.WaitKey() 
