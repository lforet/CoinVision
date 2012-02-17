#!/usr/bin/env python

import cv
#from opencv import cv2
from SimpleCV import *
import sys
import numpy as np
from coin_tools import *

sample_size = 68

if __name__=="__main__":

	if len(sys.argv) < 2:
		print "******* Requires 3 image files of the same size."
		print "This program will return the angle at which the second is in relation to the first. ***"
		sys.exit(-1)

	try:
		img1 = cv.LoadImage(sys.argv[1],cv.CV_LOAD_IMAGE_GRAYSCALE)
		img2 = cv.LoadImage(sys.argv[2],cv.CV_LOAD_IMAGE_GRAYSCALE)
		#img3 = cv.LoadImage(sys.argv[3],cv.CV_LOAD_IMAGE_GRAYSCALE)
	except:
		print "******* Could not open image files *******"
		sys.exit(-1)

	
	cv.ShowImage("Coin 1", img1)
	cv.MoveWindow ('Coin 1',50 ,50 )
	cv.ShowImage("Coin 2", img2)
	cv.MoveWindow ('Coin 2', (50 + (1 * (cv.GetSize(img1)[0]))) , 50)
	font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)
	cv.PutText(img1, "TEST", (100,100), font ,cv.RGB(255,255,255) )
	cv.ShowImage("Coin 1", img1)
	cv.WaitKey()

	coin1_center_crop, coin2_center_crop  = get_scaled_crops(img1, img2, sample_size)

	cv.ShowImage("Crop Center of Coin1", coin1_center_crop)
	cv.MoveWindow ('Crop Center of Coin1', (125 + (0 * cv.GetSize(coin1_center_crop)[1])) , (125 + (0 * cv.GetSize(coin1_center_crop)[0])) )
	cv.ShowImage("Crop Center of Coin2", coin2_center_crop)
	cv.MoveWindow ('Crop Center of Coin2', (125 + (1 * cv.GetSize(coin1_center_crop)[1])), (125 + (0 * cv.GetSize(coin2_center_crop)[0])) )

	cv.WaitKey()

	print "Finding center of crop coin 1....."
	coin1_center = (cv.GetSize(coin1_center_crop)[0]/2, cv.GetSize(coin1_center_crop)[1]/2)
	#coin1_center = find_center_of_coin(coin1_center_crop)
	print "center of crop coin 1.....", coin1_center
	print "Finding center of crop coin 2....."
	coin2_center = (cv.GetSize(coin2_center_crop)[0]/2, cv.GetSize(coin2_center_crop)[1]/2)
	#coin1_center = find_center_of_coin(coin1_center_crop)
	print "center of crop coin 2.....", coin2_center

	crop_copy1 = CVtoPIL(coin1_center_crop)
	crop_copy2 = CVtoPIL(coin2_center_crop)
	crop_copy1_width = crop_copy1.size[0]
	crop_copy2_width = crop_copy2.size[0]
	crop_copy1_height =  crop_copy1.size[1]
	crop_copy2_height =  crop_copy2.size[1]
	box1 = ( (crop_copy1_width/4) ,(crop_copy1_height/10),(crop_copy1_width-crop_copy1_width/4), (crop_copy1_height/4))
	box2 = ( (crop_copy2_width/4) ,(crop_copy2_height/10),(crop_copy2_width-crop_copy2_width/4), (crop_copy2_height/4))
	img1_roi = crop_copy1.crop(box1)
	img2_roi = crop_copy2.crop(box2)
	img1_roi = PILtoCV(img1_roi)	
	img2_roi = PILtoCV(img2_roi)

	cv.ShowImage("ROI Coin1", img1_roi)
	cv.MoveWindow ('ROI Coin1', (125 + (0 * cv.GetSize(img1_roi)[0])), (125 + (0 * cv.GetSize(img1_roi)[1])) )
	cv.ShowImage("ROI Coin2", img2_roi)
	cv.MoveWindow ('ROI Coin2', (125 + (1 * cv.GetSize(img2_roi)[0])), (125 + (0 * cv.GetSize(img2_roi)[1])) )

	cv.WaitKey()

	##############################################
	#equalizing brightness
	print "equalizing brightness of both ROI"
	img1_roi = CVtoPIL(img1_roi)	
	img2_roi = CVtoPIL(img2_roi)
	im_stat1 = ImageStat.Stat(img1_roi)
	img1_roi_mean = im_stat1.mean
	im_stat2 = ImageStat.Stat(img2_roi)
	img2_roi_mean = im_stat2.mean
	#print "img1_copy_mean:",img1_copy_mean,"  img2_copy_mean:", img2_copy_mean
	mean_ratio = img1_roi_mean[0] / img2_roi_mean[0]
	#print "mean_ratio:", mean_ratio 
	#cv.WaitKey()
	enh = ImageEnhance.Brightness(img2_roi) 
	img2_roi = enh.enhance(mean_ratio)

	img1_roi = PILtoCV(img1_roi)	
	img2_roi = PILtoCV(img2_roi)
	cv.ShowImage("ROI Coin1 Bright", img1_roi)
	cv.MoveWindow ('ROI Coin1 Bright', (125 + (0 * cv.GetSize(img1_roi)[0])), 125+(1 * cv.GetSize(img1_roi)[1]) +25)
	cv.ShowImage("ROI Coin2 Bright", img2_roi)
	cv.MoveWindow ('ROI Coin2 Bright', (125 + (1 * cv.GetSize(img2_roi)[0])), 125+(1 * cv.GetSize(img2_roi)[1]) +25)
	cv.WaitKey()

	##########################################
	# Contour
	x = 180
	#cv.Canny(img1_roi , img1_roi  ,cv.Round((x/2)),x, 3)
	#cv.Canny(img2_roi , img2_roi  ,cv.Round((x/2)),x, 3)
	#cv.Smooth(img1_roi, img1_roi, cv.CV_GAUSSIAN,3, 3)
	#cv.Smooth(img2_roi, img2_roi, cv.CV_GAUSSIAN,3, 3)
	img1_roi = CVtoPIL(img1_roi)	
	img2_roi = CVtoPIL(img2_roi)
	
	#img1_roi = img1_roi.filter(ImageFilter.CONTOUR)
	#img2_roi = img2_roi.filter(ImageFilter.CONTOUR)
	#img1_roi = img1_roi.filter(ImageFilter.FIND_EDGES)
	#img2_roi = img2_roi.filter(ImageFilter.FIND_EDGES)
	#img1_roi = img1_roi.filter(ImageFilter.EMBOSS)
	#img2_roi = img2_roi.filter(ImageFilter.EMBOSS)
	#img1_roi = img1_roi.filter(ImageFilter.SMOOTH)
	#img2_roi = img2_roi.filter(ImageFilter.SMOOTH)
	rmsdiff = rmsdiff(img1_roi, img2_roi)
	img1_roi = PILtoCV(img1_roi)	
	img2_roi = PILtoCV(img2_roi)
	cv.ShowImage("ROI Coin1 CONTOUR", img1_roi)
	cv.MoveWindow ('ROI Coin1 CONTOUR', (125 + (0 * cv.GetSize(img1_roi)[0])), 125+(2 * cv.GetSize(img1_roi)[1])+(25*2))
	cv.ShowImage("ROI Coin2 CONTOUR", img2_roi)
	cv.MoveWindow ('ROI Coin2 CONTOUR', (125 + (1 * cv.GetSize(img2_roi)[0])), 125+(2 * cv.GetSize(img2_roi)[1])+(25*2))
	cv.WaitKey()

	print "Euclidean:", cv_img_distance(img1_roi, img2_roi, 'euclidean')[0]
	print "Correlation:", cv_img_distance(img1_roi, img2_roi, 'correlation')[0]
	print "hamming:", cv_img_distance(img1_roi, img2_roi, 'hamming')[0]
	print "rms", rmsdiff
