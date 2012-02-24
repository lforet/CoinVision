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
from CoinServoDriver import *


def get_new_coin(servo, dc_motor):
		servo.arm_down()
		time.sleep(.1)
		print cv.NamedWindow('Camera', cv.CV_WINDOW_AUTOSIZE)
		
		capture =  cv.CreateCameraCapture(1)
		#time.sleep(.05)
		cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 320)
		cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
		for i in range(0, 25, 1):
			base_frame = cv.QueryFrame(capture)
			cv.ShowImage('Camera', base_frame)
			cv.WaitKey(5)
			time.sleep(.01)
		base_sum = cv.Sum(base_frame)
		base_mean = ((base_sum[0] + base_sum[1] + base_sum[2])/ ((3*base_frame.height) * (3*base_frame.width)))
		print "base_mean:", base_mean
		new_coin = False
		print 'CoinID Motor Driver Comm OPEN:', dc_motor.isOpen()
		print 'Connected to: ', dc_motor.portstr
	
		while not new_coin:
			for i in range(0, 18, 1):
				frame = cv.QueryFrame(capture)
				cv.ShowImage('Camera', frame)
				cv.WaitKey(5)
				#time.sleep(.02)
			current_sum = cv.Sum(frame)
			current_mean = ((current_sum[0] + current_sum[1] + current_sum[2])/ ((3*frame.height) * (3*frame.width)))
			print "current_mean", current_mean
			result = math.fabs(current_mean - base_mean) 
			print "dif of means:", result
			if result > 3:
				print "New coin...", result
				sys.stdout.write('\a') #beep
				new_coin = True
			#if new_coin == False: time.sleep(2)
			if new_coin == False: move_motor(dc_motor, "F", 15)
			if new_coin == False: time.sleep(.5)
			motor_stop(dc_motor)
			if new_coin == False: time.sleep(.8)
		
def move_motor(dc_motor, direction, speed):
	if direction == "F":
		cmd_str = direction + str(speed) + '%\r'
		print cmd_str
		dc_motor.write ('GO\r')
		time.sleep(.01)
		dc_motor.write (cmd_str)
		time.sleep(.01)
		dc_motor.write ('GO\r')
		time.sleep(.01)

def motor_stop(dc_motor):
	dc_motor.write ('X\r\n')


#######################   Globals
sample_size = 60


if __name__=="__main__":

 
	
	dc_motor = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
	time.sleep(1)
	coinid_servo = CoinServoDriver()
	
	#for i in range(0,5,1):
	#get_new_coin(coinid_servo, dc_motor)
	time.sleep(1)
	#	coinid_servo.arm_up(100)
	#	time.sleep(.2)
	#	coinid_servo.arm_down()
	#	time.sleep(1)
	#frame = grab_frame(1)
	#img1 = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
	#img1 = CVtoGray(frame)	
	#cv.SaveImage("images/head_1.jpg", frame)
	
	dc_motor.close()
	#cv.WaitKey()
	#sys.exit(-1)	
	

#	if len(sys.argv) < 4:
#		print "******* Requires 3 image files of the same size."
#		print "This program will return the angle at which the second is in relation to the first. ***"
#		sys.exit(-1)

	try:
		#img1 = cv.LoadImage(sys.argv[1],cv.CV_LOAD_IMAGE_GRAYSCALE)
		frame = grab_frame(1)
		img1 = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
		img1 = CVtoGray(frame)
		#cv.WaitKey()
		#img1 = CV_enhance_edge(img1)
		#cv.WaitKey()
		img2 = cv.LoadImage(sys.argv[1],cv.CV_LOAD_IMAGE_GRAYSCALE)
		img3 = cv.LoadImage(sys.argv[2],cv.CV_LOAD_IMAGE_GRAYSCALE)
	except:
		print "******* Could not open image files *******"
		sys.exit(-1)


	#print img1, img2, img3
	#print cv_img_distance(img1, img2, 'euclidean')[0]
	#cv.WaitKey()
	#sys.exit(-1)

	cv.ShowImage("Coin 1", img1)
	cv.MoveWindow ('Coin 1',50 ,50 )
	cv.ShowImage("Coin 2", img2)
	cv.MoveWindow ('Coin 2', (50 + (1 * (cv.GetSize(img1)[0]))) , 50)
	cv.ShowImage("Coin 3", img3)
	cv.MoveWindow ('Coin 3', 375, 325)


	best_dif = 0
	found_coin2 = 0
	found_coin3 = 0
	best_size = 0
	coin1_center = ((0, 0), 0)
	coin2_center = ((0, 0), 0)
	coin3_center = ((0, 0), 0)
	scaled_img_center = ((0, 0), 0)
	#print coin1_center[1]
	cv.WaitKey()
	
	#for sample_size in range(10, 70, 10):
	################## compare to coin2
	coin1_center_crop, coin2_center_crop  = get_scaled_crops(img1, img2, sample_size)

	if coin1_center[1] == 0: coin1_center = find_center_of_coin(img1)
	#cv.Circle(img1, coin1_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
	#cv.ShowImage("Coin 1", img1)
	if coin2_center[1] == 0: coin2_center = find_center_of_coin(img2)
	#cv.Circle(img2, coin2_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
	#cv.ShowImage("Coin 2", img2)
	cv.ShowImage("Crop Center of Scaled Coin1", coin1_center_crop)
	cv.MoveWindow ('Crop Center of Scaled Coin1', 100, 100)
	cv.ShowImage("Crop Center of Coin2", coin2_center_crop)
	cv.MoveWindow ('Crop Center of Coin2', 100, (125 + (cv.GetSize(coin2_center_crop)[0])) )

	#c1  = compare_images_rotation(scaled_img_center_crop, coin2_center_crop)
	#c1 = compare_images_canny(coin1_center_crop, coin2_center_crop, sample_size)
	c1 = compare_images_lbp(coin1_center_crop, coin2_center_crop)
	#c1 = compare_images_laplace(coin1_center_crop, coin2_center_crop)
	#c1 = compare_images_brightness(coin1_center_crop, coin2_center_crop)
	#c1 = compare_images_stddev(scaled_img_center_crop, coin2_center_crop)
	#c1 = compare_images_var(scaled_img_center_crop, coin2_center_crop)
	#c1 = compare_images_hu(coin1_center_crop, coin2_center_crop, sample_size)
	print "comarison coin1-> coin2:", c1


	#cv.WaitKey()
	################## compare to coin3
	coin1_center_crop, coin3_center_crop  = get_scaled_crops(img1, img3, sample_size)

	#if coin1_center[1] == 0: coin1_center = find_center_of_coin(img1)
	#cv.Circle(img1, coin1_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
	#cv.ShowImage("Coin 1", img1)
	#if coin2_center[1] == 0: coin2_center = find_center_of_coin(img2)
	#cv.Circle(img2, coin2_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
	#cv.ShowImage("Coin 2", img2)

	cv.ShowImage("Crop Center of Scaled Coin1", coin1_center_crop)
	cv.MoveWindow ('Crop Center of Scaled Coin1', 100, 100)
	cv.ShowImage("Crop Center of Coin3", coin3_center_crop)
	cv.MoveWindow ('Crop Center of Coin3', 100, (125 + 2*(cv.GetSize(coin3_center_crop)[0])) )

	#cv.WaitKey()
	#c2  = compare_images_rotation(scaled_img_center_crop, coin3_center_crop)
	#c2 = compare_images_canny(coin1_center_crop, coin3_center_crop, sample_size)
	c2 = compare_images_lbp(coin1_center_crop, coin3_center_crop)
	#c2 = compare_images_laplace(coin1_center_crop, coin3_center_crop)
	#c2 = compare_images_brightness(coin1_center_crop, coin3_center_crop)
	#c2 = compare_images_stddev(scaled_img_center_crop, coin3_center_crop)
	#c2 = compare_images_var(scaled_img_center_crop, coin3_center_crop) 
	#c2 = compare_images_hu(coin1_center_crop, coin3_center_crop, sample_size)
	print "comarison coin1-> coin3:", c2
	print

	if c1 < c2: 
	#if c1 > c2:
		found_coin2 = found_coin2 + 1
		print ; print "------------------------------------------"
		print "coin 1 is more like Coin2", found_coin2, "   ", found_coin3

	else:
		found_coin3 = found_coin3 + 1
		print ; print "------------------------------------------"
		print "Coin 1 is more like Coin3", found_coin3, "   ", found_coin2
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

	print "get oprientation of correct side.."
	if found_coin2 > found_coin3:
		correct_side_coin = cv.CloneImage(img2)

	if found_coin3 > found_coin2:
		correct_side_coin = cv.CloneImage(img3)

	
	coin1_center = find_center_of_coin(img1)
	print "center of coin 1.....", coin1_center
	
	correct_side_coin_center = find_center_of_coin(correct_side_coin)
	print "center of correct_side_coin.....", correct_side_coin_center
	scaled_img = cv.CloneImage(img1)
	#print "scaled_img=", scaled_img

	scaled_img = correct_scale(img1, correct_side_coin, coin1_center, correct_side_coin_center)
	#print "scaled_img=", scaled_img
	scaled_img_center = find_center_of_coin(scaled_img)
	#crop out center of coin based on found center
	print "Cropping center of original and scaled corrected images..."
	scaled_img_center_crop = center_crop(scaled_img, scaled_img_center, sample_size)
	cv.ShowImage("Crop Center of Scaled Coin1", scaled_img_center_crop)
	cv.MoveWindow ('Crop Center of Scaled Coin1', 100, 100)
	#cv.WaitKey()

	correct_side_coin_center_crop = center_crop(correct_side_coin, correct_side_coin_center, sample_size)
	cv.ShowImage("Crop Center of correct_side_coin_center_crop", correct_side_coin_center_crop)
	cv.MoveWindow ('Crop Center of correct_side_coin_center_crop', 100, (125 + (cv.GetSize(correct_side_coin_center_crop)[0])) )
	#cv.WaitKey()
	degrees = get_orientation_sobel( correct_side_coin_center_crop, scaled_img_center_crop, sample_size)
	print "sobel degrees:", degrees
	img1_copy = rotate_image(img1, degrees)
	cv.ShowImage("Coin 1 Oriented", img1_copy)
	cv.MoveWindow ('Coin 1 Oriented', (50 + (2 * (cv.GetSize(img1)[0]))) , 50)
	#cv.WaitKey()
	#degrees = get_orientation_canny (  coin2_center_crop, scaled_img_center_crop)
	#degrees = get_orientation_canny (  correct_side_coin_center_crop, scaled_img_center_crop)
	#print "canny degrees:", degrees
	#img1_copy = rotate_image(img1, degrees)
	cv.DestroyWindow("Coin 1")
	cv.DestroyWindow("Coin 2")
	cv.DestroyWindow("Coin 1 Oriented")
	cv.ShowImage("Coin 1", img1)
	cv.MoveWindow ('Coin 1', (50 + (0 * (cv.GetSize(img1)[0]))) , 50)
	cv.ShowImage("Coin 2", correct_side_coin)
	cv.MoveWindow ('Coin 2', (50 + (1 * (cv.GetSize(img1)[0]))) , 50)
	cv.ShowImage("Coin 1 Oriented", img1_copy)
	cv.MoveWindow ('Coin 1 Oriented', 375, 325)
	cv.WaitKey()






