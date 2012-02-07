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
from opencv import adaptors
import ImageFilter



def surf_dif(img1, img2):
	#only features with a keypoint.hessian > 600 will be extracted
	#using extended descriptors (1) -> 128 elements each
	#surfParams = cvSURFParams(600, 1)
	#gray images for detecting
	object1 = cv.CreateImage((img1.width,img1.height), 8, 1)
	cv.CvtColor(img1, object1, cv.CV_BGR2GRAY)
	object2 = cv.CreateImage((img2.width,img2.height), 8, 1)
	cv.CvtColor(img2, object2, cv.CV_BGR2GRAY)

	keypoints1, descriptors1 = cv.ExtractSURF(object1, None, (0, 400, 3, 4))
	keypoints2, descriptors2 = cv.ExtractSURF(object2, None, (0, 400, 3, 4))

	print "found %d keypoints for img1"%keypoints1.rows
	print "found %d keypoints for img2"%keypoints2.rows

	#feature matching
	ft = cv.CreateKDTree(descriptors1)
	indices, distances = cv.FindFeatures(ft, descriptors2, 1, 250)
	cv.cvReleaseFeatureTree(ft)

	#the C max value for a long (no limit in python)
	DBL_MAX = 1.7976931348623158e+308
	reverseLookup = [-1]*keypoints1.rows
	reverseLookupDist = [DBL_MAX]*keypoints1.rows

	matchCount = 0
	for j in xrange(keypoints2.rows):
	  i = indices[j]
	  d = distances[j]
	  if d < reverseLookupDist[i]:
		   if reverseLookupDist[i] == DBL_MAX:
		       matchCount+=1
		   reverseLookup[i] = j
		   reverseLookupDist[i] = d
		  
	print "found %d putative correspondences"%matchCount

	points1 = cv.CreateMat(1,matchCount,cv.CV_32FC2)
	points2 = cv.CreateMat(1,matchCount,cv.CV_32FC2)
	m=0
	for j in xrange(keypoints2.rows):
	  i = indices[j]
	  if j == reverseLookup[i]:
		   pt1 = keypoints1[i][0], keypoints1[i][1]
		   pt2 = keypoints2[j][0], keypoints2[j][1]
		   points1[m]=cv.cvScalar(pt1[0], pt1[1])
		   points2[m]=cv.cvScalar(pt2[0], pt2[1])
		   m+=1

	#remove outliers with fundamental matrix:
	status = cv.CreateMat(points1.rows, points1.cols, cv.CV_8UC1)
	fund = cv.CreateMat(3, 3, CV_32FC1)
	cv.FindFundamentalMat(points1, points2, fund, cv.CV_FM_LMEDS, 1.0, 0.99, status)
	print "fundamental matrix:"
	print fund

	print "number of outliers detected using the fundamental matrix: ", len([stat for stat in status if not stat])

	#updating the points without the outliers
	points1 = [pt for i, pt in enumerate(points1) if status[i]]
	points2 = [pt for i, pt in enumerate(points2) if status[i]]

	print "final number of correspondences:",len(points1) 





def flatten(x):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def get_SURF_points(img):
	temp_img = cv.CloneMat(img)
	keypoints = []
	try:
		storage = cv.CreateMemStorage() 
		(keypoints, descriptors) = cv.ExtractSURF(temp_img , None, storage , (0, 400, 3, 4))
		for ((xx, yy), laplacian, size, dir, hessian) in keypoints:
			print "count= %d x=%d y=%d laplacian=%d size=%d dir=%f hessian=%f" % (len(keypoints), xx, yy, laplacian, size, dir, hessian)
			cv.Circle(temp_img, (xx,yy), size, (255,0,0),1, cv.CV_AA , 0)
	except Exception, e:
    		print e

	cv.ShowImage('SURF', temp_img )
	cv.WaitKey()
	cv.DestroyWindow('SURF')
	if len(keypoints) > 0: 
		return keypoints
	else:
		return -1



def resize_img(original_img, scale_percentage):
		print original_img.height, original_img.width, original_img.nChannels
		#resized_img = cv.CreateMat(original_img.rows * scale_percentage , original.cols * scale_percenta, cv.CV_8UC3)
		resized_img = cv.CreateImage((cv.Round(original_img.width * scale_percentage) , cv.Round(original_img.height * scale_percentage)), original_img.depth, original_img.nChannels)
		cv.Resize(original_img, resized_img)
		return resized_img
		#cv.ShowImage("original_img", original_img)
		#cv.ShowImage("resized_img", resized_img)
		#cv.WaitKey()

def PILtoCV(PIL_img):
	cv_img = cv.CreateImageHeader(PIL_img.size, cv.IPL_DEPTH_8U, 1)
	cv.SetData(cv_img, PIL_img.tostring())
	return cv_img

def CVtoPIL(img):
	"""converts CV image to PIL image"""
	pil_img = Image.fromstring("L", cv.GetSize(img), img.tostring())
	cv_img = cv.CreateMatHeader(cv.GetSize(img)[1], cv.GetSize(img)[0], cv.CV_8UC1)
	cv.SetData(cv_img, pil_img.tostring())
	return pil_img

def center_crop(img, center, crop_size):
	#crop out center of coin based on found center
	x,y = center[0][0], center[0][1]
	#radius = center[1]
	radius = (crop_size * 4)
	center_crop_topleft = (x-(radius-crop_size), y-(radius-crop_size))
	center_crop_bottomright = (x+(radius-crop_size), y+(radius-crop_size))
	#print "crop top left:     ", center_crop_topleft
	#print "crop bottom right: ", center_crop_bottomright
	center_crop = cv.GetSubRect(img, (center_crop_topleft[0], center_crop_topleft[1] , (center_crop_bottomright[0] - center_crop_topleft[0]), (center_crop_bottomright[1] - center_crop_topleft[1])  ))
	#cv.ShowImage("Crop Center of Coin", center_crop)
	#cv.WaitKey()
	return center_crop


def find_center_of_coin(img):
	#create storage fo circle data
	storage = cv.CreateMat(50, 1, cv.CV_32FC3)
	#storage = cv.CreateMemStorage(0)
	cv.SetZero(storage)
	#img_copy = cv.CreateImage((img.width, img.height)), original_img.depth, img.nChannels)
	img_copy = cv.CloneImage(img)
	edges = cv.CreateImage(cv.GetSize(img), 8, 1)
	#print edges, img
	cv.Smooth(img , edges , cv.CV_GAUSSIAN,3, 3)
	#cv.Canny(edges, edges, 50, 100, 3)
	#cv.Smooth(edges, edges, cv.CV_GAUSSIAN, 3, 3)
	img_copy2 = cv.CloneImage(img_copy)
	#cv.ShowImage("grayed center image", edges)
	#cv.WaitKey()
	best_circle = ((0,0),0)
	#minRadius = 10; maxRadius = img.height
	canny = 175; param2 = 1;
	#for minRadius in range ((img.height/4), (img.height/2), 10):
	for minRadius in range (100, 190, 10):
		img_copy = cv.CloneImage(img_copy2)
		#for maxRadius in range ((img.height/2)+50, img.height, 10):
		for maxRadius in range (190, 260, 10):
			#print "minRadius: ", minRadius, " maxRadius: ", maxRadius
			circles = cv.HoughCircles(edges, storage, cv.CV_HOUGH_GRADIENT, 1, img.height, canny, param2, minRadius, maxRadius)
			
			if storage.rows > 0:
				for i in range(0, storage.rows):
					#print "Center: X:", best_circle[0][0], " Y: ", best_circle[0][1], " Radius: ", best_circle[1], " minRadius: ", minRadius, " maxRadius: ", maxRadius
					cv.WaitKey(5)
					time.sleep(.01)
					center = int(np.asarray(storage)[i][0][0]), int(np.asarray(storage)[i][0][1])
					radius = int(np.asarray(storage)[i][0][2])
					#print center, radius
					cv.Circle(img_copy, center, radius, cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 )
					cv.Circle(img_copy, center, 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
					cv.ShowImage("Center of Coin", img_copy)
					cv.MoveWindow ('Center of Coin', 50 , (50 + (1 * (cv.GetSize(img_copy)[0]))))
					if (radius > best_circle[1]) & (radius > 150) & (radius < img.height/1.5):
						best_circle = (center, radius)
						print "Found Best Circle---Center: X:", best_circle[0][0], " Y: ", best_circle[0][1], " Radius: ", best_circle[1], " minRadius: ", minRadius, " maxRadius: ", maxRadius

	return best_circle

def rmsdiff(img1, img2):
    "Calculate the root-mean-square difference between two images"
    diff = ImageChops.difference(img1, img2)
    h = diff.histogram()
    sq = (value*(idx**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(img1.size[0] * img1.size[1]))
    return rms

def get_orientation_PIL1(img1, img2): 
	best_rmsdiff = 99999999
	best_orientation = 0
	#diplay images
	img1_cv = PILtoCV(img1_pil)
	cv.ShowImage("PIL 1", img1_cv)
	cv.MoveWindow ("PIL 1",500,100)
	print 'Starting to find best orientation'
	for i in range(1, 360):
		#temp_img = rotate_image(img2, i)
		temp_img = img2.rotate(i)
		#result_img = ImageChops.difference(img1, img2)	
		#ImageChops.subtract(image1, image2, scale, offset) => image
		result = rmsdiff(img1, temp_img)

		#diplay images
		img2_cv = PILtoCV(temp_img)
		cv.ShowImage("PIL 2", img2_cv)
		cv.MoveWindow ("PIL 2", 500, 600)
		if result < best_rmsdiff: 
			best_rmsdiff = result
			best_orientation = i
			print i, "result = ", result, "  best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break
		time.sleep(.05)
	print 'Finished finding best orientation'
	return (best_orientation)


def get_orientation_SURF(img1, img2): 

	subtracted_image = cv.CreateImage(cv.GetSize(img1), 8, 1)
	temp_img = cv.CreateImage(cv.GetSize(img1), 8, 1)

	best_sum = 0
	best_orientation = 0
	print 'Starting to find best orientation using SURF'
	img1_SURFpoints = np.array(flatten(get_SURF_points(cv.GetMat(img1))))

	for i in range(1, 360):
		temp_img = rotate_image(img2, i)
		#cv.And(img1, temp_img , subtracted_image)
		img2_SURFpoints = np.array(flatten(get_SURF_points(cv.GetMat(temp_img))))
		cv.ShowImage("Image of Interest", temp_img )
		cv.MoveWindow ("Image of Interest", (100 + 2*cv.GetSize(img1)[0]), 100)

		print "img1_SURFpoints.size, img2_SURFpoints.size: ", img1_SURFpoints.size, img2_SURFpoints.size
		cv.WaitKey()
		print  img1_SURFpoints
		#surf_dist = dist(img1_SURFpoints,img2_SURFpoints)
		#print 'surf_dist =', surf_dist
		print "print scipy.spatial.distance.euclidean = ", scipy.spatial.distance.euclidean(img1_SURFpoints, img2_SURFpoints)
		#cv.ShowImage("Subtracted_Image", subtracted_image)
		#cv.MoveWindow ("Subtracted_Image", (100 + 2*cv.GetSize(img1)[0]), (150 + cv.GetSize(img1)[1]) )
		#sum_of_SURF = cv.Sum(subtracted_image)
		#if best_sum == 0: best_sum = sum_of_and[0]
		#if sum_of_and[0] > best_sum: 
		#	best_sum = sum_of_and[0]
		#	best_orientation = i
		#	print i, "Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break
		time.sleep(.1)
	print 'Finished finding best orientation'
	return (best_orientation)

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
		cv.MoveWindow ("Image of Interest", (100 + 2*cv.GetSize(img1)[0]), 100)
		cv.ShowImage("Subtracted_Image", subtracted_image)
		cv.MoveWindow ("Subtracted_Image", (100 + 2*cv.GetSize(img1)[0]), (150 + cv.GetSize(img1)[1]) )
		sum_of_and = cv.Sum(subtracted_image)
		if best_sum == 0: best_sum = sum_of_and[0]
		if sum_of_and[0] > best_sum: 
			best_sum = sum_of_and[0]
			best_orientation = i
			print i, "Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break
		time.sleep(.01)
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
	cv.MoveWindow ('Image 1',50 ,50 )
	cv.ShowImage("Image 2", img2)
	cv.MoveWindow ('Image 2', (50 + (1 * (cv.GetSize(img1)[0]))) , 50)
	#cv.WaitKey()

	img1_copy = cv.CloneImage(img1)
	img2_copy = cv.CloneImage(img2)

	#find center of coins
	print "Finding center of coins image1....."
	coin1_center = find_center_of_coin(img1_copy)
	print "Finding center of coins image2....."
	coin2_center = find_center_of_coin(img2_copy)
	#cv.WaitKey()

	#if first image is smaller than second
	if coin2_center[1] > coin1_center[1]:
		scale = float(coin2_center[1]) / float(coin1_center[1])
		print "Scaling image 1: ", scale,"%"
		img1_copy = resize_img(img1, scale)	
		img2_copy = img2
		print "Finding Center of Scaled Corrected Image 1..."
		coin1_center = find_center_of_coin(img1_copy)
		#temp_img = SimpleCV.Image(sys.argv[1]).toGray()

	#if second image is smaller than first	
	if coin2_center[1] < coin1_center[1]:
		scale = float(coin1_center[1]) / float(coin2_center[1])
		print "Scaling image 2: ", scale, "%"
		img2_copy = resize_img(img2, scale)
		img1_copy = img1
		print "Finding Center of Scaled Corrected Image 2..."
		coin2_center = find_center_of_coin(img2_copy)
	
		#temp_img = SimpleCV.Image(sys.argv[2]).toGray()

	#find center of coins after rescaling
	#print "Finding center of both coins after rescaling....."
	#coin1_center = find_center_of_coin(img1_copy)
	#coin2_center = find_center_of_coin(img2_copy)

	"""
		print "Image 2 must be scaled:", scale, "%"
		scaled_img = temp_img.scale(scale)
		#scaled_img = scaled_img.grayscale()
		scaled_img = scaled_img.getBitmap()
		cv.ShowImage("Scale Correct Image", scaled_img)
		temp_gray = cv.CreateImage(cv.GetSize(scaled_img), 8, 1)
		cv.CvtColor(scaled_img, temp_gray, cv.CV_RGB2GRAY)
		temp_gray_copy = cv.CloneImage(temp_gray)
	"""
	#cv.ShowImage("Image 1_copy", img1_copy)
	#cv.ShowImage("Image 2_copy", img2_copy)
	#cv.WaitKey()
	#sys.exit(-1)


	#crop out center of coin based on found center
	print "Cropping center of original and scaled corrected images..."
	coin1_center_crop = center_crop(img1_copy, coin1_center, 70)
	cv.ShowImage("Crop Center of Coin1", coin1_center_crop)
	cv.MoveWindow ('Crop Center of Coin1', 100, 100)
	#cv.WaitKey()
	coin2_center_crop = center_crop(img2_copy, coin2_center, 70)
	cv.ShowImage("Crop Center of Coin2", coin2_center_crop)
	cv.MoveWindow ('Crop Center of Coin2', 100, (125 + (cv.GetSize(coin1_center_crop)[0])) )
	#cv.WaitKey()

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


	
	#print "mats?:", coin1_center_crop, coin2_center_crop
	#print "these are the cropped images: ", img1_copy, img2_copy
	cv.WaitKey()


	#for i in range(50, 300, 50):
	i=190
	img1_copy = cv.CloneMat(coin1_center_crop) 
	img2_copy = cv.CloneMat(coin2_center_crop)
	img1_pil = CVtoPIL(img1_copy)
	img2_pil = CVtoPIL(img2_copy)
	img1_pil = ImageOps.equalize(img1_pil) 
	img2_pil = ImageOps.equalize(img2_pil)
	img1_copy = PILtoCV(img1_pil)
	img2_copy = PILtoCV(img2_pil)
	print "Equalizing the histograms..."
	cv.ShowImage("Equalized Image 1_copy", img1_copy)
	cv.MoveWindow ('Equalized Image 1_copy', (101 + (1 * (cv.GetSize(coin1_center_crop)[0]))) , 100)
	cv.ShowImage("Equalized Image 2_copy", img2_copy)
	cv.MoveWindow ("Equalized Image 2_copy", (101 + (1 * (cv.GetSize(coin1_center_crop)[0]))) , (155 + (cv.GetSize(coin1_center_crop)[0])) )
	cv.WaitKey()
	#time.sleep(2)

	#cv.Erode(img1_copy, img1_copy , element=None, iterations=1)
	#cv.Erode(img2_copy, img2_copy , element=None, iterations=1)
	cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
	cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN, 3, 3)
	cv.Canny(img1_copy ,img1_copy  ,cv.Round((i/2)),i, 3)
	cv.Canny(img2_copy, img2_copy  ,cv.Round((i/2)),i, 3)
	#cv.Laplace(img1_copy, img1_copy)
	#cv.Laplace(img2_copy, img2_copy)

	#cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
	#cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN, 3, 3)
	#cv.Erode(img1_copy, img1_copy , element=None, iterations=1)
	#cv.Erode(img2_copy, img2_copy , element=None, iterations=1)

	cv.ShowImage  ("Canny Coin 1", img1_copy )
	cv.MoveWindow ('Canny Coin 1', (101 + (1 * (cv.GetSize(coin1_center_crop)[0]))) , 100)
	cv.ShowImage  ("Canny Coin 2", img2_copy )
	cv.MoveWindow ('Canny Coin 2', (101 + (1 * (cv.GetSize(coin1_center_crop)[0]))) , (125 + (cv.GetSize(coin1_center_crop)[0])) )
	print "Press any key to find correct orientation"  
	#cv.WaitKey()
	degrees = get_orientation(img1_copy, img2_copy)
	print "Degrees Re-oriented: ", degrees
	img3 = cv.CloneMat(coin2_center_crop)
	img3 = rotate_image(coin2_center_crop, degrees)
	cv.ShowImage("Orientation Corrected Image2", img3 )
	cv.MoveWindow ("Orientation Corrected Image2", 100, 800)
	print "i=", i
	cv.WaitKey() 

	"""
	#pil orientation
	img1_copy = cv.CloneMat(coin1_center_crop) 
	img2_copy = cv.CloneMat(coin2_center_crop)
	img1_pil = CVtoPIL(img1_copy)
	img2_pil = CVtoPIL(img2_copy)
	img1_pil = ImageOps.equalize(img1_pil) 
	img2_pil = ImageOps.equalize(img2_pil)
	#img1_copy = PILtoCV(img1_pil)
	#img2_copy = PILtoCV(img2_pil)

	img1_pil = img1_pil.filter(ImageFilter.EDGE_ENHANCE)
	img2_pil = img2_pil.filter(ImageFilter.EDGE_ENHANCE)
	img1_pil = img1_pil.filter(ImageFilter.SMOOTH)
	img2_pil = img2_pil.filter(ImageFilter.SMOOTH)	
	#img1_pil = img1_pil.filter(ImageFilter.EMBOSS)
	#img2_pil = img2_pil.filter(ImageFilter.EMBOSS)		
	img1_pil = img1_pil.filter(ImageFilter.FIND_EDGES)
	img2_pil = img2_pil.filter(ImageFilter.FIND_EDGES)
	img1_pil = img1_pil.filter(ImageFilter.SMOOTH)
	img2_pil = img2_pil.filter(ImageFilter.SMOOTH)
	#img1_pil = img1_pil.filter(ImageFilter.CONTOUR)
	#img2_pil = img2_pil.filter(ImageFilter.CONTOUR)
	#img1_pil = img1_pil.filter(ImageFilter.DETAIL)
	#img2_pil = img2_pil.filter(ImageFilter.DETAIL)

	#filtHorizontal = [1, 0, -1, 2, 0, -2, 1, 0, -1]
	#filtVertical   = [1, 2, 1, 0, 0, 0, -1, -2, -1]

	#img1_pil = img1_pil.filter(ImageFilter.BLUR)
	#img1_pil = img1_pil.filter(ImageFilter.BLUR)
	#edgeHorizontal = im.filter((3,3), filtHorizontal)
	#edgeVertical = im.filter((3,3), filtVertical)
	#img1_pil = img1_pil.filter( (3,3) , filtHorizontal)

	#img2_pil = img2_pil.filter( (3,3) , filtHorizontal)
	#edgeVertical = im.filter((3,3), filtVertical)

	#BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, and SHARPEN. 



	degrees = get_orientation_PIL1(img1_pil, img2_pil)
	print "Degrees Re-oriented: ", degrees
	img3 = cv.CloneMat(coin2_center_crop)	
	img3 = rotate_image(coin2_center_crop, degrees)
	cv.ShowImage("PIL Orientation Corrected Image2", img3 )
	cv.MoveWindow ("PIL Orientation Corrected Image2", 600, 800)
	#print "i=", i
	cv.WaitKey() 
	"""

	### compare using surf
	img1_copy = cv.CloneMat(coin1_center_crop) 
	img2_copy = cv.CloneMat(coin2_center_crop)
	print "Using SURF"
	cv.WaitKey() 
	degrees = get_orientation_SURF(img1_copy, img2_copy)
	print "Degrees Re-oriented: ", degrees
	cv.WaitKey() 	


