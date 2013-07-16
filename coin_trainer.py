import sys
sys.path.append( "../lib/" )

import easygui as eg
from img_processing_tools import *
from PIL import ImageStat, Image, ImageDraw
import cv, cv2
import time
import math
import mahotas
from mahotas.features import surf
import numpy as np
import cPickle as pickle
import csv
import milk
from threading import *
from pylab import *
import scipy.spatial
from CoinServoDriver import *
from coin_tools import *
import glob
import pylab
#from SimpleCV import *
import itertools
from skimage.feature import hog
from skimage.feature import daisy
#from skimage import data
#import matplotlib.pyplot as plt
from skimage import data, color, exposure
from skimage.feature import match_template

#for Structural SIMilarity
import numpy
import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi


#import matplotlib.pyplot as plt


def get_new_coin(servo, dc_motor):
	servo.arm_down()
	base_frame = snap_shot(1)
	#time.sleep(1)
	new_coin = False
	print 'CoinID Motor Driver Comm OPEN:', dc_motor.isOpen()
	print 'Connected to: ', dc_motor.portstr
	pilimg1 = CVtoPIL(CVtoGray(base_frame))
	print "pilimg1 = ", pilimg1
	while not new_coin:
		if new_coin == False: move_motor(dc_motor, "F", 20)
		if new_coin == False: time.sleep(.5)
		motor_stop(dc_motor)
		if new_coin == False: time.sleep(.8)
		frame = snap_shot(1)
		pilimg2 = CVtoPIL(CVtoGray(frame))
		rms_dif = rmsdiff(pilimg1, pilimg2)
		print "RMS Dif:", rms_dif 
		if rms_dif > 20:
			print "New coin...", rms_dif
			sys.stdout.write('\a') #beep
			new_coin = True

		
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

def snap_shot(usb_device):
	print "snapshot called"
	#capture from camera at location 0
	now = time.time()
	webcam1 = None
	frame = None
	#try:	
	while webcam1 == None:
		webcam1 = cv2.VideoCapture(usb_device)
		#webcam1 = cv.CreateCameraCapture(usb_device)
		#time.sleep(.05)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
		time.sleep(.1)
	for i in range(6):
		ret, frame = webcam1.read()
		frame = array2cv(frame)
		#cv.GrabFrame(webcam1)
		#frame = cv.QueryFrame(webcam1)
	#except:
	#	print "******* Could not open WEBCAM *******"
	#	print "Unexpected error:", sys.exc_info()[0]
		#raise		
		#sys.exit(-1)
	#print frame
	#print webcam1
	#while webcam1 != None:
	cv2.VideoCapture(usb_device).release()
	#print webcam1
	#time.sleep(1)
	#print webcam1
	return frame
 

def display_image(img_filename, wait_time):
	global ready_to_display
	while ready_to_display != True:
		time.sleep(1)
		#print "waiting"
	#time.sleep(wait_time)
	img = imread(img_filename)
	#img = CVtoPIL(array2cv(img))
	#img = img.transpose(1)
	#img = img.transpose(2)
	#img.save("pil.png")
	pylab.ion()	
	#print "a:", a
	pylab.imshow(img)
	pylab.draw()

'''
The function to compute SSIM
@param param: img_mat_1 1st 2D matrix
@param param: img_mat_2 2nd 2D matrix
'''
def compute_ssim(img_mat_1, img_mat_2):
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel=numpy.zeros((gaussian_kernel_width,gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1=img_mat_1.astype(numpy.float)
    img_mat_2=img_mat_2.astype(numpy.float)
    
    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)
        
    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)
    
    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12;
    
    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2
    
    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=numpy.average(ssim_map)

    return index



def template_matching(img_to_match, database_img):

	img_to_match_cropped_coin_only = preprocess_img(img_to_match)
	database_img_cropped_coin_only = preprocess_img(database_img)
	img_to_match_coin_center = find_center_of_coin(img_to_match_cropped_coin_only)
	print "Coin only Center of img_to_match_cropped_coin_only:", img_to_match_coin_center
	database_img_coin_center = find_center_of_coin(database_img_cropped_coin_only)
	print "Coin only Center of database_img_cropped_coin_only:",database_img_coin_center
	img_to_match_final_cropped = cv2array(center_crop(img_to_match_cropped_coin_only, img_to_match_coin_center, 40))
	database_img_final_cropped = cv2array(center_crop(database_img_cropped_coin_only, database_img_coin_center, 40))



	result = match_template(image, coin)
	ij = np.unravel_index(np.argmax(result), result.shape)
	x, y = ij[::-1]
	print x,y
	sys.exit(-1)


def preprocess_houghlines (img, num_lines):

	temp_img = cv2array(preprocess_img(img))
	#print img, temp_img
	USE_STANDARD = True
	x = 140
	if USE_STANDARD: x = 200

	lines = np.array([[[]]])
	while len(lines[0]) < num_lines:
		try:
			edges = cv2.Canny(temp_img, (int(x/2)), x , apertureSize=3)
			if USE_STANDARD: 
				lines = cv2.HoughLines(edges, 1, math.pi/180,num_lines)
			else:
				lines = cv2.HoughLinesP(edges, 1, math.pi/180, 40, None, 40, 10);
			if lines == None: 
				lines = np.array([[[]]])	
			x = x -2
		except:
			x = x -2
		#time.sleep(.05)
	#cv2.imwrite("houghlines_canny.png", edges)
		print "canny threshold: ", x , " Lines: ", len(lines[0])
	cv2.imwrite("houghlines_canny_center_cropped.png",edges)
	#temp_top_lines = lines[0][:num_lines]
	#top_lines = []

	coin_center = ( (int(edges.shape[0]/2),int(edges.shape[1]/2)), edges.shape[0])
	cropped = cv2array(center_crop(array2cv(edges), coin_center, 40))
	cv2.imwrite("houghlines_canny_center_cropped2.png",cropped)
	return cropped




	
def houghlines(img, num_lines):
	"""
	Python: cv2.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines
	Parameters:	
	image - 8-bit, single-channel binary source image. The image may be modified by the function.
	lines - Output vector of lines. Each line is represented by a 4-element vector, where and are the ending points of each detected line segment.
	rho - Distance resolution of the accumulator in pixels.
	theta - Angle resolution of the accumulator in radians.
	threshold - Accumulator threshold parameter. Only those lines are returned that get enough votes (  ).
	minLineLength - Minimum line length. Line segments shorter than that are rejected.
	maxLineGap - Maximum allowed gap between points on the same line to link them.
	"""
	USE_STANDARD = True
	x = 140
	if USE_STANDARD: x = 480

	lines = np.array([[[]]])
	while len(lines[0]) < num_lines:
		try:
			edges = cv2.Canny(img, (int(x/2)), x , apertureSize=3)
			if USE_STANDARD: 
				lines = cv2.HoughLines(edges, 1, math.pi/180,num_lines)
			else:
				lines = cv2.HoughLinesP(edges, 1, math.pi/180, 40, None, 40, 10);
			if lines == None: 
				lines = np.array([[[]]])	
			x = x -2
		except:
			x = x -2
	time.sleep(.5)
	cv2.imwrite("houghlines_canny.png", edges)
	print "x: ", x , " Lines: ", len(lines[0])
	temp_top_lines = lines[0][:num_lines]
	top_lines = []

	coin_center = ( (int(edges.shape[0]/2),int(edges.shape[1]/2)), edges.shape[0])
	cropped = cv2array(center_crop(array2cv(edges), coin_center, 10))
	print cropped.shape, np.sum(cropped)

	#get surf features
	#features_surf = surf.surf(edges)
	#features_surf = surf.surf(np.mean(img,2))
	#print "SURF:", features_surf, " len:", len(features_surf)
	#raw_input("Press Enter to continue...")
	#sorted_features_surf = sort(features_surf[:25])
	#print "SORTED SURF:", sorted_features_surf
	#raw_input("Press Enter to continue...")
	#return [np.sum(cropped)]
	#return sorted_features_surf.flatten()
	
	sys.exit(-1)

	if USE_STANDARD:
		# for houghlines proper
		top_lines = np.asarray(top_lines)
		############# Line sort descending (longest to shortest)
		top_lines = temp_top_lines[temp_top_lines[:,0].argsort()][::-1]#.flatten()
		#top_lines = top_lines[:int(num_lines/2)]
		top_lines = top_lines[:5]
		print "STANDARD LINES sorted:", top_lines 
		#sys.exit(-1)	

	else:
		##for houghlineP
		for line in temp_top_lines:
			dist = scipy.spatial.distance.cdist(([[line[0],line[1]]]), ([[line[2], line[3]]]), 'euclidean')
			top_lines.append([line[0],line[1], line[2], line[3], dist[0][0]])
		top_lines = np.asarray(top_lines)
		############# Line sort descending (longest to shortest)
		top_lines = top_lines[top_lines[:,4].argsort()][::-1]#.flatten()
		#top_lines = top_lines[:int(num_lines/2)]
		top_lines = top_lines[:5]
		print "PROB LINES sorted:", top_lines
	
	#hough lines comparison
	#theta = atan( (double)(pt2.y - pt1.y)/(pt2.x - pt1.x) ); /*slope of line*/
    #degree = theta*180/CV_PI; 

	features_to_return = []
	
	###### Draw lines on temp img
	temp_img = img
	if USE_STANDARD:
		for (rho, theta) in top_lines:
			a = cos(theta)
			b = sin(theta)
			x0 = a * rho 
			y0 = b * rho
			degree = theta*180/math.pi;
			pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
			pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
			#if pt2[1] > 380 and pt2[1] < 550:
			cv2.line(temp_img, pt1, pt2, (0,0,255), 2)
			print degree, rho, theta, pt1, pt2, a, b, x0, y0	
			features_to_return.append(degree)
	else:
		for line in top_lines:
			#print line, line[0]
			pt1 = (int(line[0]),int(line[1]))
			pt2 = (int(line[2]),int(line[3]))
			cv2.line(temp_img, pt1, pt2, (0,0,255), 2)
	cv2.imwrite("houghlines_canny.png", edges)
	cv2.imwrite("houghlines.png", temp_img)
	#sys.exit(-1)

	#raw_input("Press Enter to continue...")
	return np.array(features_to_return).flatten()
	
	'''
	###distance from center
	center_pt = np.array([int(img.shape[0]/2),int(img.shape[1]/2)])
	center_dist = []
	for point in top_lines:
		line_pt1 =  np.array([point[0], point[1]])
		line_pt2 =  np.array([point[2], point[3]])
		pt1_dist = scipy.spatial.distance.cdist([center_pt], [line_pt1], 'euclidean')
		pt2_dist = scipy.spatial.distance.cdist([center_pt], [line_pt2], 'euclidean')
		center_dist.append([pt1_dist[0][0], pt2_dist[0][0]])
	center_dist = np.array([center_dist]).flatten()
	
	
	#avg_feature_histo = np.histogram(avg_features, bins=20)[0]
	avg_feature_sum =np.sum(center_dist )
	avg_feature_std =np.std(center_dist )
	avg_feature_median =np.median(center_dist )
	avg_feature_mean =	np.mean(center_dist )
	avg_feature_var =  np.var(center_dist )
	#print "histo:", avg_feature_histo, len(avg_feature_histo)
	print "center_dist:" , center_dist
	print "dist SUM:", avg_feature_sum
	print "dist STD:", avg_feature_std 
	print "dist MEDIAN:", avg_feature_median
	print "dist MEAN:", avg_feature_mean 
	print "dist VAR:", avg_feature_var 
	
	center_dist = np.append(center_dist,avg_feature_sum)
	center_dist = np.append(center_dist,avg_feature_std)
	center_dist = np.append(center_dist,avg_feature_median)
	center_dist = np.append(center_dist,avg_feature_mean)
	center_dist = np.append(center_dist,avg_feature_var)
	print center_dist, len(center_dist)
	
	features_to_return = []
	features_to_return.append([avg_feature_sum, avg_feature_std, avg_feature_median, avg_feature_mean, avg_feature_var])
	#print "to return:", np.array(features_to_return).flatten()
	#raw_input("Press Enter to continue...")
	#return np.array(features_to_return).flatten()
	#return  (center_dist[0], center_dist[1])
	####distance from 1st 3 pts
	'''
	'''
	x = np.array([top_lines[0][0], top_lines[0][1]])
	y = np.array([top_lines[0][2], top_lines[0][3]])
	z = np.array([top_lines[1][0], top_lines[1][1]])

	total_lines_point_dist = []
	for point in top_lines:
		line_pt1 =  np.array([point[0], point[1]])
		line_pt2 =  np.array([point[2], point[3]])
		pt1_x_dist = scipy.spatial.distance.cdist([x], [line_pt1], 'euclidean')
		pt1_y_dist = scipy.spatial.distance.cdist([y], [line_pt1], 'euclidean')
		pt1_z_dist = scipy.spatial.distance.cdist([z], [line_pt1], 'euclidean')
		
		pt2_y_dist = scipy.spatial.distance.cdist([y], [line_pt2], 'euclidean')
		pt2_z_dist = scipy.spatial.distance.cdist([z], [line_pt2], 'euclidean')
		#print line_pt1, pt1_x_dist, pt1_y_dist, pt1_z_dist, line_pt2, pt2_x_dist, pt2_y_dist, pt2_z_dist
		total_lines_point_dist.append([pt1_x_dist[0][0], pt1_y_dist[0][0], pt1_z_dist[0][0], pt2_x_dist[0][0], pt2_y_dist[0][0], pt2_z_dist[0][0]])
	total_lines_point_dist = np.array(total_lines_point_dist)#.flatten()
	
	print "total_lines_point_dist:"; print  total_lines_point_dist; print
	#print np.histogram(total_lines_point_dist, bins=20)[0]
	#time.sleep(2)
	#sys.exit(-1)
	'''
	#if USE_STANDARD:
	#return top_lines.flatten()
	#else:		
	#return total_lines_point_dist.flatten()
	
	


def features360(img, preprocess=True, coin_center=None, step360=360, averaging=False, classID=0):
	if preprocess == True:
		cropped_coin_only = preprocess_img(img)
	else:
		cropped_coin_only = img
	if coin_center == None:
		coin_center = find_center_of_coin(cropped_coin_only)
		print "Coin only Center of Coin", coin_center
	#sys.exit(-1)
	totals_array = []
	for x in xrange(0, 360, step360):
		rotated_img = rotate_image(cropped_coin_only,x)
		cropped = cv2array(center_crop(rotated_img, coin_center, 40))
		cv2.imwrite("rotated.png", cropped)
		print type(cropped)
		features = find_features(cropped)

		#features = houghlines(cropped, 50)
		#raw_input("Press Enter to continue...")
		#features = mahotas.features.haralick(cropped).mean(0)
		#features = mahotas.features.tas(cropped)
		#hog
		#cropped = cropped.reshape(240,240)
		#features = hog(cropped, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3), visualise=False, normalise=True)
		#hogimage = hogimage.reshape(240,240,1)
		#features = mahotas.features.lbp(cropped, 1, 8)
		#features = cropped.flatten()
		#print len(features)
		#time.sleep(5)
		print "Degree:", x#, "   Totals:", features
		features_to_return = []	
		if averaging == False:	
			features_to_return = features
			#totals_array.append(features[0])
			#totals_array.append(features[1])
			#print "totals_array:", totals_array
			if classID != 0 and len(features_to_return) > 0 : save_data(features_to_return, classID)		
		############ Averaging		
		if averaging == True:
			if len(totals_array) != 0:
				totals_array = np.sum([totals_array,features], axis = 0)
			else:
				totals_array = features
			
			#sys.exit(-1)
			if x > 0:
				xdiv = int (x / step360)
				div_array = []
				#Build divisor array (number of features to get avg)
				#print features
				for i in range( len(features)):
					div_array.append(xdiv)
				#print div_array
				avg_features = np.divide(totals_array,div_array).flatten()
				print "avg features:", avg_features
				#print scipy.spatial.distance.cdist([features.flatten()], [avg_features.flatten()], 'euclidean')
				avg_feature_histo = np.histogram(avg_features, bins=20)[0]
				avg_feature_sum =np.sum(avg_features.flatten())
				avg_feature_std =np.std(avg_features.flatten())
				avg_feature_median =np.median(avg_features.flatten())
				avg_feature_mean =	np.mean(avg_features.flatten())
				avg_feature_var =  np.var(avg_features.flatten())
				print "histo:", avg_feature_histo, len(avg_feature_histo)
				print "dist SUM:", avg_feature_sum
				print "dist STD:", avg_feature_std 
				print "dist MEDIAN:", avg_feature_median
				print "dist MEAN:", avg_feature_mean 
				print "dist VAR:", avg_feature_var 
	if averaging == True:
		for i in avg_features:
			#print i, avg_feature_histo
			features_to_return.append(i)
		for i in avg_feature_histo:
			#print i, avg_feature_histo
			features_to_return.append(i)
		features_to_return.extend([avg_feature_sum,avg_feature_std, avg_feature_median, avg_feature_mean , avg_feature_var])

	#features_to_return = totals_array

	print "features_to_return:", features_to_return, len(features)
	#sys.exit(-1)
	if classID != 0 and averaging == True: save_data(features_to_return, classID)	
	return features_to_return


def preprocess_img(img1):
	print "Greying image"
	grey = array2cv(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
	print "Smoothing Image"
	cv.Smooth(grey,grey,cv.CV_GAUSSIAN,3,3)
	print "Finding Center of Coin"
	coin_center = find_center_of_coin(grey)
	print "Center of Coin:", coin_center
	z = 3
	while True:
		try:
			diameter = 	int(coin_center[1]/z)
			print "trying cropped coin diameter:",  diameter
			cropped_coin_only = center_crop(grey, coin_center, diameter)
			break
		except:
			z = z + .1
	print "Final cropped coin diameter: ",  diameter
	#cv2.imwrite("cropped_coin.png", cv2array(cropped_coin_only))

	#########################################
	#		Display Results
	#######################
	#display =Thread(target=display_image, args=("cropped.png",.05,))
	#display.daemon=True
	#display.start()
	print "Finished preprocessing..."
	return cropped_coin_only
	#return cropped

def binary_compare(img):
	#print img, type(img), img.shape
	#time.sleep(5)
	#print img[0]
	img = resize_img(array2cv(img), .25)
	cv2.imwrite("postprocessed_img.png", cv2array(img))
	features = []
	features = flatten(cv2array(img))
	return features


def goodfeatures(img):
	#print type(img)
	#img = array2cv(preprocess_img(img))
	features = cv2.goodFeaturesToTrack(img, maxCorners=50, qualityLevel=0.1, minDistance=10)
	features = features[:40]
	return features.flatten()

def find_features(img):
	#img = preprocess_img(img)
	#features = houghlines(img, 20)
	#features = features360_avg(img)
	#features = features360(img, preprocess=True, coin_center=None, step360=1, averaging=False, classID=0)
	#features = binary_compare(img)
	#features = goodfeatures(img)
	#print img, type(img)

	#gray scale the image if neccessary
	#if img.shape[2] == 3:
	#	img = img.mean(2)

	#img = mahotas.imread(imname, as_grey=True)
	#features = mahotas.features.haralick(img).mean(0)
	#f2 = features
	#print 'haralick features:', features, len(features), type(features[0])
	
	#features = mahotas.features.lbp(img, 1, 8)
	#f2 = np.concatenate((f2,features))
	#print 'LBP features:', features, len(features), type(features[0])

	#features = mahotas.features.tas(img)
	#f2 = np.concatenate((f2,features))
	#print 'TAS features:', features, len(features), type(features[0])


	#features = mahotas.features.zernike_moments(np.mean(img,2), 2, degree=8)
	#print 'ZERNIKE features:', features, len(features), type(features[0])
	#f2 = np.concatenate((f2,features))

	#hu_moments = []
	#hu_moments =  np.array(cv.GetHuMoments(cv.Moments(cv.fromarray(img))))
	#print "HU_MOMENTS: ", hu_moments
	#features = flatten(hu_moments)
	#f2 = np.concatenate((f2,features))
	#features = f2


	#DAISY
	#gray scale the image if neccessary
	if img.shape[2] != None:
		img = img.mean(2)

	img_step = int(img.shape[1]/4)
	img_radius = int(img.shape[1]/10)
	descs, descs_img = daisy(img, step=img_step, radius=img_radius, rings=2, histograms=8, orientations=8, normalization='l2', visualize=True)
	features = descs.ravel()
	#plt.axis('off')
	#plt.imshow(descs_img)
	#descs_num = descs.shape[0] * descs.shape[1]
	#plt.title('%i DAISY descriptors extracted:' % descs_num)
	#plt.show()
	#print len(features.ravel())


	#print len(features[0][0])
	#print "All Features: ", features, len(features)
	'''
	#features_surf = surf.surf(np.mean(img,2))
	#print "SURF:", features_surf, " len:", len(features_surf)

	try:
		import milk

		# spoints includes both the detection information (such as the position
		# and the scale) as well as the descriptor (i.e., what the area around
		# the point looks like). We only want to use the descriptor for
		# clustering. The descriptor starts at position 5:
		descrs = features_surf[:,5:]

		# We use 5 colours just because if it was much larger, then the colours
		# would look too similar in the output.
		k = 5
		surf_pts_to_ID = 50
		values, _  = milk.kmeans(descrs, k)
		colors = np.array([(255-52*i,25+52*i,37**i % 101) for i in xrange(k)])
	except:
		values = np.zeros(100)
		colors = [(255,0,0)]
	surf_img = surf.show_surf(np.mean(img,2), features_surf[:surf_pts_to_ID], values, colors)
	#imshow(surf_img)
	#show()
	'''
	#houghlines opencv

	#try:
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#gray = CVtoGray(numpy2CV(img))
	#print gray

	#except:
	#	print "no houghlines available"
	#img1 = mahotas.imread('temp.png')
	#T_otsu = mahotas.thresholding.otsu(img1)
	#seeds,_ = mahotas.label(img > T_otsu)
	#labeled = mahotas.cwatershed(img1.max() - img1, seeds)
	#imshow(labeled)
	#show()
	'''


	for x in hu_moments[0]:
		if x < 0: x = (x * -1)
		print math.log10(x)
	distmin = 0
	degree = 0
	for x in range(359):

		img2 = cv.CloneImage(array2cv(grey))
		#img2 = rotate_image(img2, x)
		#print type(img2)
		img2 = CVtoPIL(img2)
		img2 = img2.rotate(x, expand=1)
		#print type(img2)
		img2 = PILtoCV(img2,1)
		cv.ShowImage("45", img2)
		cv.WaitKey()
		#print type(img2)
		hu_moments2 = []
		hu_moments2 =  np.array(cv.GetHuMoments(cv.Moments(cv.GetMat(img2))))
		hu_moments2 = hu_moments2.reshape(1, (hu_moments2.shape[0]))
		distance_btw_images = scipy.spatial.distance.cdist(hu_moments, hu_moments2,'euclidean')
		if (distance_btw_images < distmin): degree = x
		print x, ": ", log10(distance_btw_images )
		#print "HUMOMENTS2: ", hu_moments2
		#for x in hu_moments2:
		#	print math.log10(x)
	print "degree = ", degree
	'''

	return features

def classify(model, features):
     return model.apply(features)

def grab_frame_from_video(video):
	frame = video.read()
	return frame


def predict_class_360(img, step360=360):
	cropped_coin_only = preprocess_img(img)
	coin_center = find_center_of_coin(cropped_coin_only)
	print "Coin only Center of Coin", coin_center
	#sys.exit(-1)
	classID_votes = [0,0,0,0]
	#model = pickle.load( open( "coinvision_ai_model.mdl", "rb" ) )
	for x in xrange(0, 360, step360):
		rotated_img = rotate_image(cropped_coin_only,x)
		cropped = cv2array(center_crop(rotated_img, coin_center, 40))
		cv2.imwrite("rotated.png", cropped)
		#features = features360(rotated_img, preprocess=False,coin_center=coin_center, step360=360, averaging=False, classID=0)
		classID =  predict_class(cropped)
		if classID == 1: answer = "Jefferson HEADS"
		if classID == 2: answer = "Monticello TAILS"
		if classID == 3: answer = "Other HEADS"
		if classID == 4: answer = "Other TAILS"
		print "predicted classID:", answer
		classID_votes[classID-1] = classID_votes[classID-1] +1
		print "classID_votes:", classID_votes, classID_votes.index(max(classID_votes))
		#time.sleep(1)
	final_classID_vote = classID_votes.index(max(classID_votes)) + 1
	'''
	from sklearn import svm
	model = pickle.load( open( "coinvision_ai_model_svc.mdl", "rb" ) )
	print model.predict(features)

	from sklearn.neighbors import KNeighborsClassifier
	#neigh = KNeighborsClassifier(n_neighbors=3)
	neigh= pickle.load( open( "coinvision_ai_model_knn.mdl", "rb" ) )
	print neigh.predict(features)
	#print neigh.predict_proba(1)
	'''

	#eg.msgbox("predicted classID:"+answer)
	return final_classID_vote



def predict_class(img):
	features = find_features(img)
	classID = 0

	from sklearn import svm
	model_svm = pickle.load( open( "coinvision_ai_model_svc.mdl", "rb" ) )
	classID_svm = model_svm.predict(features)
	print "SVM predicted classID:", classID_svm
	#print "SVM predicted prob:", model_svm.predict_proba(features)

	from sklearn.neighbors import KNeighborsClassifier
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh = pickle.load( open( "coinvision_ai_model_knn.mdl", "rb" ) )
	print "KNN predicted classID:", neigh.predict(features)
	print "KNN predicted prob:", neigh.predict_proba(features)
	
	from sklearn.svm import LinearSVC
	clf = LinearSVC()
	clf = pickle.load( open( "coinvision_ai_model_lr.mdl", "rb" ) )
	print "LR predicted classID:", clf.predict(features)
	#print "lr predicted prob:", clf.predict_proba(features)

	from sklearn.linear_model import LogisticRegression
	clf2 = pickle.load(open( "coinvision_ai_model_clf2.mdl", "rb" ) )
	print "LR2 predicted classID:",clf2.predict(features)
	print "LR2 predicted prob:", clf2.predict_proba(features)
#try:
	model = pickle.load( open( "coinvision_ai_model.mdl", "rb" ) )
	classID = classify(model, features)	
	print "classID: = ", classID
	if classID == 1: answer = "Jefferson HEADS"
	if classID == 2: answer = "Monticello TAILS"
	if classID == 3: answer = "Other HEADS"
	if classID == 4: answer = "Other TAILS"
	print "predicted classID:", answer
	#eg.msgbox("predicted classID:"+answer)
	return classID_svm
#except:
	print "could not predict...bad data"


def save_data(features, classID):
	data_filename = 'coinvision_feature_data.csv'
	###########################
	print 'writing image features to file: ', data_filename
	# delete data file and write header
	#f_handle = open(data_filename, 'w')
	#f_handle.write(str("classid, lbp, i3_histogram, rgb_histogram, sum_I3, sum2_I3, median_I3, avg_I3, var_I3, stddev_I3, rms_I3"))
	#f_handle.write('\n')
	#f_handle.close()

	#write class data to file
	f_handle = open(data_filename, 'a')
	f_handle.write(str(classID))
	f_handle.write(', ')
	f_handle.close()

	f_handle = open(data_filename, 'a')
	for i in range(len(features)):
		f_handle.write(str(features[i]))
		f_handle.write(" ")
	f_handle.write('\n')
	f_handle.close()




def process_all_images():
	path = "../coin_images/"
	#print path+'jheads/*.jpg'
	steps = 360
	for name in glob.glob(path+'jheads/*.jpg'):
		classID = "1"
		print name
		img = cv2.imread(name)		
		#features = find_features(img)
		#save_data(features, classID)
		features360(img, step360=steps, averaging=False, classID=1)

	for name in glob.glob(path+'jtails/*.jpg'):
		classID = "2"
		print name
		img = cv2.imread(name)
		#img = preprocess_img(img)
		#features = find_features(img)		
		#save_data(features, classID)
		features360(img, step360=steps, averaging=False, classID=2)

	for name in glob.glob(path+'oheads/*.jpg'):
		classID = "3"
		print name
		img = cv2.imread(name)
		#img = preprocess_img(img)
		#features = find_features(img)
		#save_data(features, classID)
		features360(img, step360=steps, averaging=False, classID=3)

	for name in glob.glob(path+'otails/*.jpg'):
		classID = "4"
		print name
		img = cv2.imread(name)
		#img = preprocess_img(img)
		#features = find_features(img)
		#save_data(features, classID)
		features360(img, step360=steps, averaging=False, classID=4)

def train_ai():
			
			data = []
			classID = []
			features = []
			features_temp_array = []
			
			'''
			#SIMPLECV
			#bows
			feature_extractors = []
			extractor_names = []
			# Training data set paths for classification(suppervised learnning)
			image_dirs = ['../coin_images/jheads/',
						  '../coin_images/jtails/',
						  '../coin_images/oheads/',
						  '../coin_images/otails/',
						  ]
			# Different class labels for multi class classification
			class_names = ['jhead','jtail','ohead', 'otail']
			
			
			#preprocess all training images
			for directory in image_dirs:
				for filename in glob.glob(directory + '/*.jpg'):
					print "Processing:", filename
					img = cv2.imread(filename)
					temp_img  = preprocess_houghlines (img, 100)
					temp_str = filename.rsplit('/')
					temp_str = temp_str[len(temp_str)-1]
					temp_str = directory + '/temp/' + temp_str
					print temp_str
					cv2.imwrite(temp_str, temp_img)
					#raw_input('press enter to continue : ')
			#sys.exit(-1)
			
			
			#build array of directories for bow
			#image_dirs2 = []
			#for directory in image_dirs:
			#	image_dirs2.append(directory + '/temp/')
			#print image_dirs2

			# Different class labels for multi class classification
			extractor_name = 'hue'
			if extractor_name == 'bow':
				feature_extractor = BOFFeatureExtractor() # feature extrator for bag of words methodology
				feature_extractor.generate(image_dirs,imgs_per_dir=40) # code book generation
			elif extractor_name == 'hue':
				feature_extractor = HueHistogramFeatureExtractor()
			elif extractor_name == 'morphology':
				feature_extractor = MorphologyFeatureExtractor()
			elif extractor_name == 'haar':
				feature_extractor = HaarLikeFeatureExtractor()
			elif extractor_name == 'edge':
				feature_extractor = EdgeHistogramFeatureExtractor()
			image_dirs2 = image_dirs
			#bow_features = BOFFeatureExtractor()
			#bow_features.generate(image_dirs2,imgs_per_dir=40, verbose=True) # code book generation
			#bow_features.generate(image_dirs2,imgs_per_dir=200,numcodes=256,sz=(11,11),img_layout=(16,16),padding=4 )
			#bow_features.save('codebook.png','bow.txt')

			#print "extractor_names:", extractor_names, feature_extractors
			# initializing classifier with appropriate feature extractors list
			#print type(bow_features), bow_features, bow_features.getFieldNames(), bow_features.getNumFields()
			#raw_input('bow saved...Enter : ')
			#bow_features = None
			
			#bow_features = BOFFeatureExtractor()
			#print type(bow_features), bow_features, bow_features.getFieldNames(), bow_features.getNumFields()
			#bow_features.load('bow.txt')
			#print type(bow_features), bow_features, bow_features.getFieldNames(), bow_features.getNumFields()
			feature_extractors.append(feature_extractor)
			#raw_input('bow loaded Enter : ')

			#extractor_names.append(extractor_name)
			
			classifier_name = 'naive'
			if classifier_name == 'naive':
				classifier = NaiveBayesClassifier(feature_extractors)
			elif classifier_name == 'svm':
				classifier = SVMClassifier(feature_extractors)
			elif classifier_name == 'knn':
				classifier = KNNClassifier(feature_extractors, 2)
			elif classifier_name == 'tree':
				classifier = TreeClassifier(feature_extractors)

			# train the classifier to generate hypothesis function for classification
			#print "image_dirs:", image_dirs, class_names
			classifier.train(image_dirs2,class_names,disp=None,savedata='features.txt',verbose=True)
			
			print 'classifier:', type(classifier), classifier
			raw_input('press enter to continue :')
			#pickle.dump( classifier, open( "coinvision_ai_model2.mdl", "wb" ),2 )
			#classifier.save('coinvision_ai_model.mdl')
			print 'classifier:', type(classifier), classifier
			#classifier = NaiveBayesClassifier.load('coinvision_ai_model.mdl')

			#raw_input('press enter to continue : let me try loading bow file')
			#classifier2 = NaiveBayesClassifier.load('coinvision_ai_model.mdl')
			#classifier2.setFeatureExtractors(feature_extractors)
			#print 'classifier2:', type(classifier2), classifier2
			#classifier.load("coinvision_ai_model.mdl")
			#classifier2.load('coinvision_ai_model.mdl')
			#print 'classifier:', type(classifier2), classifier2
			raw_input('press enter to continue : ')
			print 'testing ai:'
			test_images_path = "../coin_images/unclassified"
			extension = "*.jpg"

			if not test_images_path:
				path = os.getcwd() #get the current directory
			else:
				path = test_images_path

			directory = os.path.join(path, extension)
			files = glob.glob(directory)

			count = 0 # counting the total number of training images
			error = 0 # conuting the total number of misclassification by the trained classifier
			for image_file in files:
				new_image = Image(image_file)
				category = classifier.classify(new_image)
				print "image_file:", image_file + "     classified as: " + category
				if image_file[-9] == 't':
					if category == 'jhead' or category == 'ohead':
						print "INCORRECT CLASSIFICATION"
						error += 1
				if image_file[-9] == 'h':
					if category == 'jtail' or category == 'otail':
						print "INCORRECT CLASSIFICATION"
						error += 1
				count += 1
			# reporting the results
			print ' * classifier : ', classifier
			print ' * extractors :', extractor_names
			print ' *', error, 'errors out of', count
			raw_input('edned press enter to continue : ')
			return
			'''
		#try: 
			data_filename = 'coinvision_feature_data.csv'
			print 'reading features and classID: ', data_filename
			f_handle = open(data_filename, 'r')
			reader = csv.reader(f_handle)
			#read data from file into arrays
			for row in reader:
				data.append(row)

			for row in range(0, len(data)):
				#print features[row][1]
				classID.append(int(data[row][0]))
				features_temp_array.append(data[row][1].split(" "))

			#removes ending element which is a space
			for x in range(len(features_temp_array)):
				features_temp_array[x].pop()
				features_temp_array[x].pop(0)

			#convert all strings in array to numbers
			temp_array = []
			for x in range(len(features_temp_array)):
				temp_array = [ float(s) for s in features_temp_array[x] ]
				features.append(temp_array)

			#make numpy arrays
			features = np.asarray(features)
			#print classID, features 

			
			learner = milk.defaultclassifier(mode='really-slow')
			model = learner.train(features, classID)
			pickle.dump( model, open( "coinvision_ai_model.mdl", "wb" ) )
			

		#except:
			print "could not retrain.. bad file"
			
			from sklearn import svm
			model = svm.SVC(gamma=0.001, C=100.)
			model.fit(features, classID)
			pickle.dump( model, open( "coinvision_ai_model_svc.mdl", "wb" ) )
			
			from sklearn.neighbors import KNeighborsClassifier
			neigh = KNeighborsClassifier(n_neighbors=3)
			neigh.fit(features, classID)
			pickle.dump( neigh, open( "coinvision_ai_model_knn.mdl", "wb" ) )
			
			from sklearn.svm import LinearSVC
			clf = LinearSVC()
			clf = clf.fit(features, classID)
			pickle.dump( clf, open( "coinvision_ai_model_lr.mdl", "wb" ) )
			
			from sklearn.linear_model import LogisticRegression
			clf2 = LogisticRegression().fit(features, classID)
			pickle.dump( clf2 , open( "coinvision_ai_model_clf2.mdl", "wb" ) )
		
			return 

def sift():
	
	#img1=Image("cropped.png")
	#img2=Image("temp.png")
	img1 = cv2.imread("cropped.png")
	img2 = cv2.imread('temp.png')

	'''
	#i.drawSIFTKeyPointMatch(i1,distance=50).show()
	img = cv2.imread("temp.png")
	template = cv2.imread("cropped.png")
	detector = cv2.FeatureDetector_create("SIFT")
	descriptor = cv2.DescriptorExtractor_create("SIFT")

	skp = detector.detect(img)
	skp, sd = descriptor.compute(img, skp)

	tkp = detector.detect(template)
	tkp, td = descriptor.compute(template, tkp)

	flann_params = dict(algorithm=1, trees=4)
	flann = cv2.flann_Index(sd, flann_params)
	idx, dist = flann.knnSearch(td, 1, params={})
	del flann
	
	#print idx, dist
	#sys.exit(-1)
	dist = dist[:,0]/2500.0
	dist = dist.reshape(-1,).tolist()
	idx = idx.reshape(-1).tolist()
	indices = range(len(dist))
	indices.sort(key=lambda i: dist[i])
	dist = [dist[i] for i in indices]
	idx = [idx[i] for i in indices]

	distance = 50
	skp_final = []
	for i, dis in itertools.izip(idx, dist):
		if dis < distance:
		    skp_final.append(skp[i])
		else:
		    break

	print skp_final
	'''
	#compare_images_features_points(img1, img2, 'sift')
	compare_images_features_points(img1, img2, 'surf')
	#compare_images_features_points(img1, img2, 'orb')

	return


def subsection_image(pil_img, sections, visual):
	sections = sections / 4
	#print "sections= ", sections
	fingerprint = []

	# - -----accepts image and  number of sections to divide the image into (resolution of fingerprint)
	# ---------- returns a subsectioned image classified by terrain type
	img_width, img_height = pil_img.size
	#print "image size = img_wdith= ", img_width, "  img_height=", img_height, "  sections=", sections
	#cv.DestroyAllWindows()
	#time.sleep(2)
	if visual == True:
		cv_original_img1 = PILtoCV(pil_img,3)
		#cv.NamedWindow('Original', cv.CV_WINDOW_AUTOSIZE)
		cv.ShowImage("Original",cv_original_img1 )
		#cv_original_img1_ary = np.array(PIL2array(pil_img))
		#print cv_original_img1_ary
		#cv2.imshow("Original",cv_original_img1_ary) 
		cv.MoveWindow("Original", ((img_width)+100),50)
	#pil_img = rgb2I3 (pil_img)
	#cv.WaitKey()
	#cv.DestroyWindow("Original")
	temp_img = pil_img.copy()
	xsegs = img_width  / sections
	ysegs = img_height / sections
	#print "xsegs, ysegs = ", xsegs, ysegs 
	for yy in range(0, img_height-ysegs+1 , ysegs):
		for xx in range(0, img_width-xsegs+1, xsegs):
			#print "Processing section =", xx, yy, xx+xsegs, yy+ysegs
			box = (xx, yy, xx+xsegs, yy+ysegs)
			#print "box = ", box
			cropped_img1 = pil_img.crop(box)
			I3_mean =   ImageStat.Stat(cropped_img1).mean
			I3_mean_rgb = (int(I3_mean[0]), int(I3_mean[1]), int(I3_mean[2]))
			print "I3_mean: ", I3_mean
			sub_ID = predict_class(image2array(cropped_img1))
			print "sub_ID:", sub_ID
			#fingerprint.append(sub_ID)
			if visual == True:
				cv_cropped_img1 = PILtoCV(cropped_img1,3)
				cv.ShowImage("Fingerprint",cv_cropped_img1 )
				cv.MoveWindow("Fingerprint", (img_width+100),50)
				if sub_ID == 1: I3_mean_rgb = (50,150,50)
				if sub_ID == 2: I3_mean_rgb = (150,150,150)
				if sub_ID == 3: I3_mean_rgb = (0,0,200)
				ImageDraw.Draw(pil_img).rectangle(box, (I3_mean_rgb))
				cv_img = PILtoCV(pil_img,3)
				cv.ShowImage("Image",cv_img)
				cv.MoveWindow("Image", 50,50)
				cv.WaitKey(20)
				time.sleep(.1)
				#print xx*yy
				#time.sleep(.05)
	#cv.DestroyAllWindows()
	cv.DestroyWindow("Fingerprint")
	cv.WaitKey(100)
	cv.DestroyWindow("Image")
	cv.WaitKey(100)
	cv.DestroyWindow("Original")
	cv.WaitKey(100)
	cv.DestroyWindow("Image")
	cv.WaitKey()
	time.sleep(2)
	#print "FINGERPRINT: ", fingerprint
	#cv.WaitKey()
	#return fingerprint
	return 9


#def get_scores(candidate_img, img, classid, x)

def compare_rms(img_to_classify):
	path = "../coin_images/"
	rms_classID = 0
	comp_classID = 0
	absdif_classID = 0

	absdif_score = 99999999
	comp_score =   0
	rms_score =    99999999
	degree = 0
	cannyx = 200
	crop_size = 55

	cropped_coin_only = preprocess_img(img_to_classify)
	coin_center = find_center_of_coin(cropped_coin_only)
	print "Coin only Center of Coin", coin_center
	img_to_classify_cropped = center_crop(cropped_coin_only , coin_center, crop_size)
	cv.Canny(img_to_classify_cropped,img_to_classify_cropped ,cv.Round((cannyx/2)),cannyx, 3)
	cv2.imwrite("cropped_coin.png", cv2array(img_to_classify_cropped))
	#sys.exit(-1)

	for name in glob.glob(path+'jheads/*.jpg'):
		print name
		img = cv2.imread(name)
		img_cropped = preprocess_img(img)
	
		cv.Canny(img_cropped,img_cropped ,cv.Round((cannyx/2)),cannyx, 3)
		cv2.imwrite("cropped_coin2.png", cv2array(img_cropped))

		for x in range(360):
			rotated_img = rotate_image(img_cropped  ,x)
			cropped = cv2array(center_crop(rotated_img, coin_center, crop_size))
			cv2.imwrite("rotated.png", cropped)
			'''
			temp_absdif_score = np.sum(cv2.absdiff(cropped,cv2array(img_to_classify_cropped)))
			if temp_absdif_score < absdif_score: 
				absdif_score = temp_absdif_score
				absdif_classID = "1"
				absdif_degree = x
				print "absdiff:", absdif_score,"  New classID:", absdif_classID, " Degree:", absdif_degree
			#temp_comp_score = np.sum(cv2.compare(cropped, cv2array(img_to_classify_cropped), cv2.CMP_EQ))
			'''
			img_to_classify_cropped_array = cv2array(img_to_classify_cropped)
			img_to_classify_cropped_array_reshaped = np.reshape(img_to_classify_cropped_array, (img_to_classify_cropped_array.shape[0],img_to_classify_cropped_array.shape[1]) )
			cropped_reshaped = np.reshape(cropped, (cropped.shape[0],cropped.shape[1]) )
	
			temp_comp_score = compute_ssim(cropped_reshaped, img_to_classify_cropped_array_reshaped)
			if temp_comp_score > comp_score: 
				comp_score = temp_comp_score
				comp_classID = "1"
				comp_degree = x
				print "comp_score:", comp_score,"  New classID:", comp_classID, " Degree:", comp_degree
				rotated_img2 = cv2array(rotate_image(array2cv(img_to_classify) , (360-comp_degree)))
				cv2.imwrite("comp_degree.png", rotated_img2)
			'''
			temp_rms_score = compare_images(cropped , cv2array(img_to_classify_cropped))
			if temp_rms_score[0] < rms_score:
				rms_score = temp_rms_score[0]
				rms_classID = "1"
				rms_degree = x
				print "RMS Score:", rms_score,"  New classID:", rms_classID, " Degree:", rms_degree
			'''

	for name in glob.glob(path+'jtails/*.jpg'):
		print name
		img = cv2.imread(name)
		img_cropped = preprocess_img(img)	

		cv.Canny(img_cropped,img_cropped ,cv.Round((cannyx/2)),cannyx, 3)
		cv2.imwrite("cropped_coin2.png", cv2array(img_cropped))

		for x in range(360):
			rotated_img = rotate_image(img_cropped  ,x)
			cropped = cv2array(center_crop(rotated_img, coin_center, crop_size))
			cv2.imwrite("rotated.png", cropped)

			img_to_classify_cropped_array = cv2array(img_to_classify_cropped)
			img_to_classify_cropped_array_reshaped = np.reshape(img_to_classify_cropped_array, (img_to_classify_cropped_array.shape[0],img_to_classify_cropped_array.shape[1]) )
			cropped_reshaped = np.reshape(cropped, (cropped.shape[0],cropped.shape[1]) )
	
			temp_comp_score = compute_ssim(cropped_reshaped, img_to_classify_cropped_array_reshaped)
			if temp_comp_score > comp_score: 
				comp_score = temp_comp_score
				comp_classID = "2"
				comp_degree = x
				print "comp_score:", comp_score,"  New classID:", comp_classID, " Degree:", comp_degree
				rotated_img2 = cv2array(rotate_image(array2cv(img_to_classify) , (360-comp_degree)))
				cv2.imwrite("comp_degree.png", rotated_img2)

	for name in glob.glob(path+'oheads/*.jpg'):
		print name
		img = cv2.imread(name)
		img_cropped = preprocess_img(img)

		cv.Canny(img_cropped,img_cropped ,cv.Round((cannyx/2)),cannyx, 3)
		cv2.imwrite("cropped_coin2.png", cv2array(img_cropped))

		for x in range(360):
			rotated_img = rotate_image(img_cropped  ,x)
			cropped = cv2array(center_crop(rotated_img, coin_center, crop_size))
			cv2.imwrite("rotated.png", cropped)

			temp_comp_score = np.sum(cv2.compare(cropped, cv2array(img_to_classify_cropped), cv2.CMP_EQ))
			if temp_comp_score > comp_score: 
				comp_score = temp_comp_score
				comp_classID = "3"
				comp_degree = x
				print "comp_score:", comp_score,"  New classID:", comp_classID, " Degree:", comp_degree
				rotated_img2 = cv2array(rotate_image(array2cv(img_to_classify) , (360-comp_degree)))
				cv2.imwrite("comp_degree.png", rotated_img2)


	for name in glob.glob(path+'otails/*.jpg'):
		print name
		img = cv2.imread(name)
		img_cropped = preprocess_img(img)

		cv.Canny(img_cropped,img_cropped ,cv.Round((cannyx/2)),cannyx, 3)
		cv2.imwrite("cropped_coin2.png", cv2array(img_cropped))

		for x in range(360):
			rotated_img = rotate_image(img_cropped  ,x)
			cropped = cv2array(center_crop(rotated_img, coin_center, crop_size))
			cv2.imwrite("rotated.png", cropped)

			temp_comp_score = np.sum(cv2.compare(cropped, cv2array(img_to_classify_cropped), cv2.CMP_EQ))
			if temp_comp_score > comp_score: 
				comp_score = temp_comp_score
				comp_classID = "4"
				comp_degree = x
				print "comp_score:", comp_score,"  New classID:", comp_classID, " Degree:", comp_degree
				rotated_img2 = cv2array(rotate_image(array2cv(img_to_classify) , (360-comp_degree)))
				cv2.imwrite("comp_degree.png", rotated_img2)

	#print "Final classID:", classID, " Degree:", degree
	print "ClassID:"
	print "1 = Heads"
	print "2 = Tails"
	print "3 = Other Heads"
	print "4 = Other Tails"
	print "Final comp_score:", comp_score,"  Comp classID:", comp_classID, " Degree:", comp_degree


if __name__=="__main__":
	ready_to_display = False
 	
	try:
		dc_motor = serial.Serial(port='/dev/ttyACM2', baudrate=9600, timeout=1)
		time.sleep(1)
		coinid_servo = CoinServoDriver()
		time.sleep(1)
	#dc_motor.close()
	except:
		print "no hardware (WEBCAM) attached"
		#sys.exit(-1)

	#get_new_coin(coinid_servo, dc_motor)
	#time.sleep(1)
	#coinid_servo.arm_up(100)
	#time.sleep(.2)
	#coinid_servo.arm_down()
	#time.sleep(.2)

	
	print "********************************************************************"
	print "*   must have coinvision hardware attched                          *"
	print "********************************************************************"
	video = None
	webcam1 = None
	img1 = None
	try:
		img1 = cv2.imread('temp.png')
	except:
		pass
	if len(sys.argv) > 1:
		try:
			video = cv2.VideoCapture(sys.argv[1])
			print video, sys.argv[1]
		except:
			print "******* Could not open image/video file *******"
			print "Unexpected error:", sys.exc_info()[0]
			#raise		
			sys.exit(-1)
	#eg.rootWindowPosition = "+100+100"
	reply = ""
	while True:
		
		ready_to_display = False
		#eg.rootWindowPosition = eg.rootWindowPosition
		print 'reply=', reply		

		#if reply == "": reply = "Next Frame"

		if reply == "JHEAD":
			img1 = cv2.imread('temp.png')
			path = "../coin_images/jheads/"
			filename = str(time.time()) + ".jpg"
			image_to_save = array2image(img1)
			image_to_save.save(path+filename)	

		if reply == "JTAIL":
			img1 = cv2.imread('temp.png')
			path = "../coin_images/jtails/"
			filename = str(time.time()) + ".jpg"
			image_to_save = array2image(img1)
			image_to_save.save(path+filename)	

		if reply == "OHEAD":
			img1 = cv2.imread('temp.png')
			path = "../coin_images/oheads/"
			filename = str(time.time()) + ".jpg"
			image_to_save = array2image(img1)
			image_to_save.save(path+filename)


		if reply == "OTAIL":
			img1 = cv2.imread('temp.png')
			path = "../coin_images/otails/"
			filename = str(time.time()) + ".jpg"
			image_to_save = array2image(img1)
			image_to_save.save(path+filename)

		if reply == "Test Img":	
			img1 = cv2.imread('temp.png')
			path = "../coin_images/unclassified/"
			filename = str(time.time()) + ".jpg"
			image_to_save = array2image(img1)
			image_to_save.save(path+filename)
		
		if reply == "Quit":
			print "Quitting...."
			sys.exit(-1)

		if reply == "RMS":
			#sift()
			img_to_classify = cv2.imread('temp.png')
			compare_rms(img_to_classify)

		if reply == "Predict":
			print "AI predicting"
			img1 = cv2.imread('temp.png')
			#img1 = preprocess_img(img1)
			#cv2.imwrite('postprocessed_img.png', img1)
			#predicted_classID = predict_class(img1)
			predicted_classID = predict_class_360(img1, step360=10)
			if predicted_classID == 1: answer = "Jefferson HEADS"
			if predicted_classID == 2: answer = "Monticello TAILS"
			if predicted_classID == 3: answer = "Other HEADS"
			if predicted_classID == 4: answer = "Other TAILS"
			print "------------------------------------------"
			print "FINAL: predicted_classID:", answer

		if reply == "Subsection":
			img1 = Image.open('temp.png')
			print img1
			xx = subsection_image(img1, 16,True)
			print xx
			#while (xx != 9):
			#	time.sleep(1)

		if reply == "Features":
			#img = mahotas.imread('temp.png', as_grey=True)
			img1 = cv2.imread('temp.png')
			#img1 = preprocess_img(img1)
			find_features(img1)
			ready_to_display = True


		if reply == "Retrain AI":
			print "Retraining AI"
			train_ai()

		if reply == "Next Coin":
			print "clearing coin shoot..."
			coinid_servo.arm_up(100)
			time.sleep(.2)
			coinid_servo.arm_down()
			#time.sleep(.2)
			print "Acquiring new image.."
			if video != None: 
				img1 = np.array(grab_frame_from_video(video)[1])
			else:
				get_new_coin(coinid_servo, dc_motor)
				time.sleep(.5)
				img1 = cv2array(snap_shot(1))
			#print img1
			#img1 = preprocess_img(img1)
			cv2.imwrite('temp.png', img1)
			#img1 = array2image(img1)
			#print type(img1)
			#img1.save()

		if reply == "Process Imgs":
			print "Processing all training images....."
			process_all_images()
			time.sleep(1)

		if reply == "Del AI File":
			data_filename = 'coinvision_feature_data.csv'
			f_handle = open(data_filename, 'w')
			f_handle.write('')
			f_handle.close()
			data_filename = 'coinvision_ai_model.mdl'
			f_handle = open(data_filename, 'w')
			f_handle.write('')
			f_handle.close()
		try:
			reply =	eg.buttonbox(msg='Coin Trainer', title='Coin Trainer', choices=('RMS', 'JHEAD', 'JTAIL', 'OHEAD', 'OTAIL', 'Test Img', 'Next Coin', 'Predict', 'Features','Process Imgs', 'Retrain AI' , 'Del AI File', 'Quit'), image='temp.png', root=None)
		except:
			pass



