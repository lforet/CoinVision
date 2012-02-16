#!/usr/bin/env python

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
from opencv import adaptors
import ImageFilter
from coin_tools import *
from pylab import imread, imshow, gray, mean



#######################   Globals
sample_size = 40


if __name__=="__main__":

	if len(sys.argv) < 4:
		print "******* Requires 3 image files of the same size."
		print "This program will return the angle at which the second is in relation to the first. ***"
		sys.exit(-1)

	try:
		img1 = cv.LoadImage(sys.argv[1],cv.CV_LOAD_IMAGE_GRAYSCALE)
		img2 = cv.LoadImage(sys.argv[2],cv.CV_LOAD_IMAGE_GRAYSCALE)
		img3 = cv.LoadImage(sys.argv[3],cv.CV_LOAD_IMAGE_GRAYSCALE)
	except:
		print "******* Could not open image files *******"
		sys.exit(-1)

	#print cv_img_distance(img1, img2, 'euclidean')[0]
	#cv.WaitKey()
	#sys.exit(-1)
	
	#img1_copy = cv.CloneImage(img1)
	#img2_copy = cv.CloneImage(img2)

	cv.ShowImage("Coin 1", img1)
	cv.MoveWindow ('Coin 1',50 ,50 )
	cv.ShowImage("Coin 2", img2)
	cv.MoveWindow ('Coin 2', (50 + (1 * (cv.GetSize(img1)[0]))) , 50)
	cv.ShowImage("Coin 3", img3)
	cv.MoveWindow ('Coin 3', (50 + (2 * (cv.GetSize(img1)[0]))) , 50)


	"""
	################## compare to coin1 to coin1
	scaled_img = cv.CloneImage(img1)
	#print "scaled_img=", scaled_img
	scaled_img = correct_scale(img1, img1, coin1_center, coin1_center)
	scaled_img_center = find_center_of_coin(scaled_img)
	#crop out center of coin based on found center
	print "Cropping center of original and scaled corrected images..."
	scaled_img_center_crop = center_crop(scaled_img, scaled_img_center, sample_size)
	#print "scaled_img_center_crop:", scaled_img_center_crop
	cv.ShowImage("Crop Center of Scaled Coin1", scaled_img_center_crop)
	cv.MoveWindow ('Crop Center of Scaled Coin1', 100, 100)
	#cv.WaitKey()
	c0 = compare_images_rotation(scaled_img_center_crop, scaled_img_center_crop)
	c0 = compare_images_canny(scaled_img_center_crop, scaled_img_center_crop)
	"""

	best_dif = 0
	found_coin2 = 0
	found_coin3 = 0
	best_size = 0
	coin1_center = ((0, 0), 0)
	coin2_center = ((0, 0), 0)
	coin3_center = ((0, 0), 0)
	scaled_img_center = ((0, 0), 0)
	#print coin1_center[1]
	#cv.WaitKey()
	
	for sample_size in range(10, 70, 10):
		################## compare to coin2
		print "Finding center of coin 1....."
		if coin1_center[1] == 0: coin1_center = find_center_of_coin(img1)
		print "center of coin 1.....", coin1_center
		#cv.Circle(img1, coin1_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
		cv.ShowImage("Coin 1", img1)
		print "Finding center of coin 2....."
		if coin2_center[1] == 0: coin2_center = find_center_of_coin(img2)
		print "center of coin 2.....", coin2_center
		#cv.Circle(img2, coin2_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
		cv.ShowImage("Coin 2", img2)
		if scaled_img_center[1] == 0: 
			scaled_img = correct_scale(img1, img2, coin1_center, coin2_center)
			scaled_img_center = find_center_of_coin(scaled_img)
		#crop out center of coin based on found center
		print "Cropping center of original and scaled corrected images..."
		scaled_img_center_crop = center_crop(scaled_img, scaled_img_center, sample_size)
		#print "scaled_img_center_crop:", scaled_img_center_crop
		cv.ShowImage("Crop Center of Scaled Coin1", scaled_img_center_crop)
		cv.MoveWindow ('Crop Center of Scaled Coin1', 100, 100)
		#cv.WaitKey()
		coin2_center_crop = center_crop(img2, coin2_center, sample_size)
		cv.ShowImage("Crop Center of Coin2", coin2_center_crop)
		cv.MoveWindow ('Crop Center of Coin2', 100, (125 + (cv.GetSize(coin2_center_crop)[0])) )
		#print "coin2_center_crop:", coin2_center_crop
		#cv.WaitKey()
		#print "scaled_img_center_crop, coin2_center_crop:", scaled_img_center_crop, coin2_center_crop
	
		#c1  = compare_images_rotation(scaled_img_center_crop, coin2_center_crop)
		#c1 = compare_images_canny(scaled_img_center_crop, coin2_center_crop)
		#c1 = compare_images_lbp(scaled_img_center_crop, coin2_center_crop)
		#c1 = compare_images_laplace(scaled_img_center_crop, coin2_center_crop)
		c1 = compare_images_rms(scaled_img_center_crop, coin2_center_crop)
		#c1 = compare_images_stddev(scaled_img_center_crop, coin2_center_crop)
		#c1 = compare_images_var(scaled_img_center_crop, coin2_center_crop)
		print "comarison coin1-> coin2:", c1

	
		#cv.WaitKey()
		################## compare to coin3
		print "Finding center of coin 1....."
		if coin1_center[1] == 0: coin1_center = find_center_of_coin(img1)
		print "center of coin 1.....", coin1_center
		#cv.Circle(img1, coin1_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
		cv.ShowImage("Coin 1", img1)
		print "Finding center of coin 3....."
		if coin3_center[1] == 0: coin3_center = find_center_of_coin(img3)
		print "center of coin 3.....", coin3_center
		#cv.Circle(img3, coin3_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
		cv.ShowImage("Coin 3", img3)
		if scaled_img_center[1] == 0: 
			scaled_img = correct_scale(img1, img3, coin1_center, coin3_center)
			scaled_img_center = find_center_of_coin(scaled_img)

		#crop out center of coin based on found center
		print "Cropping center of original and scaled corrected images..."
		scaled_img_center_crop = center_crop(scaled_img, scaled_img_center, sample_size)
		cv.ShowImage("Crop Center of Scaled Coin1", scaled_img_center_crop)
		cv.MoveWindow ('Crop Center of Scaled Coin1', 100, 100)
		#cv.WaitKey()
		coin3_center_crop = center_crop(img3, coin3_center, sample_size)
		cv.ShowImage("Crop Center of Coin3", coin3_center_crop)
		cv.MoveWindow ('Crop Center of Coin3', 100, (125 + 2*(cv.GetSize(coin3_center_crop)[0])) )
	
		#cv.WaitKey()
		#c2  = compare_images_rotation(scaled_img_center_crop, coin3_center_crop)
		#c2 = compare_images_canny(scaled_img_center_crop, coin3_center_crop)
		#c2 = compare_images_lbp(scaled_img_center_crop, coin3_center_crop)
		#c2 = compare_images_laplace(scaled_img_center_crop, coin3_center_crop)
		c2 = compare_images_rms(scaled_img_center_crop, coin3_center_crop)
		#c2 = compare_images_stddev(scaled_img_center_crop, coin3_center_crop)
		#c2 = compare_images_var(scaled_img_center_crop, coin3_center_crop) 
		print "comarison coin1-> coin3:", c2
		print

		if c1 < c2: 
		#if c1 > c2:
			found_coin2 = found_coin2 + 1
			print "coin 1 is more like coin2", found_coin2, "   ", found_coin3
	
		else:
			found_coin3 = found_coin3 + 1
			print "coin 1 is more like coin3", found_coin3, "   ", found_coin2
		dif = math.fabs(c1-c2)
		print "Dif = ", dif
		if dif > best_dif: 
			best_dif = dif
			best_size = sample_size
		print
		print "Best Dif:", best_dif, "  BEST SIZE =", best_size, "   sample_size:", sample_size
		time.sleep(2)
		#cv.WaitKey()

	#print "img1>img1:", c0
	#print "img1>img2:", c1_sobel
	#print "img1>img3:", c2_sobel
	#if c1_sobel < c2_sobel: 
	#	print "coin 1 is more like coin2"
	#else:
	#	print "coin 1 is more like coin3"
	#print math.fabs(c1_sobel-c1_canny)
	#print math.fabs(c2_sobel-c2_canny)
	#print "(c1_sobel-c2_sobel)=", math.fabs(c1_sobel-c2_sobel)
	#print "(c2_canny-c2_canny)=", math.fabs(c1_canny-c2_canny)

	"""
	coin2_center = find_center_of_coin(img2)
	print "center of coin 2.....", coin2_center
	scaled_img = cv.CloneImage(img1)
	#print "scaled_img=", scaled_img
	scaled_img = correct_scale(img1, img2, coin1_center, coin2_center)
	#print "scaled_img=", scaled_img
	scaled_img_center = find_center_of_coin(scaled_img)
	#crop out center of coin based on found center
	print "Cropping center of original and scaled corrected images..."
	scaled_img_center_crop = center_crop(scaled_img, scaled_img_center, sample_size)
	cv.ShowImage("Crop Center of Scaled Coin1", scaled_img_center_crop)
	cv.MoveWindow ('Crop Center of Scaled Coin1', 100, 100)
	#cv.WaitKey()
	coin2_center_crop = center_crop(img2, coin2_center, sample_size)
	cv.ShowImage("Crop Center of Coin2", coin2_center_crop)
	cv.MoveWindow ('Crop Center of Coin2', 100, (125 + (cv.GetSize(coin2_center_crop)[0])) )
	#cv.WaitKey()
	degrees = get_orientation_sobel( scaled_img_center_crop,  coin2_center_crop)
	print "sobel degrees:", degrees
	img1_copy = rotate_image(img2, degrees)
	cv.ShowImage("Coin 1 Oriented", img1_copy)
	cv.MoveWindow ('Coin 1 Oriented', (50 + (2 * (cv.GetSize(img1)[0]))) , 50)
	cv.WaitKey()
	#degrees = get_orientation_canny (  coin2_center_crop, scaled_img_center_crop)
	degrees = get_orientation_canny (  scaled_img_center_crop, coin2_center_crop, )
	print "canny degrees:", degrees
	img1_copy = rotate_image(img2, degrees)
	cv.ShowImage("Coin 1 Oriented", img1_copy)
	cv.MoveWindow ('Coin 1 Oriented', (50 + (2 * (cv.GetSize(img1)[0]))) , 50)
	cv.WaitKey()
	"""






