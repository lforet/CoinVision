#!/usr/bin/env python

#This program will find and return the center of a coin image. It assumes only one coin per image

import cv
import sys
import numpy
import Image 
import math, operator
import time
import scipy.spatial
import ImageChops


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


def dist(x,y):   
    return numpy.sqrt(numpy.sum((x-y)**2))

def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"
    h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))

def get_orientation(img1, img2):

	subtracted_image = cv.CreateImage(cv.GetSize(img1), 8, 1)
	temp_img = cv.CreateImage(cv.GetSize(img1), 8, 1)

	best_sum = 0
	best_orientation = 0
	for i in range(1, 360):
		temp_img = rotate_image(img2, i)
		cv.And(img1, temp_img , subtracted_image)
		cv.ShowImage("subtracted_image", subtracted_image)
		cv.ShowImage("Image of Interest", temp_img )
		sum_of_and = cv.Sum(subtracted_image)
		if best_sum == 0: best_sum = sum_of_and[0]
		if sum_of_and[0] > best_sum: 
			best_sum = sum_of_and[0]
			best_orientation = i
		print i, "Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "  best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break
		time.sleep(.05)
	return (best_orientation)





if __name__=="__main__":
	
	if len(sys.argv) < 3:
		print "******* Requires 2 image files. This program will find image2 inside image1. *******"
		sys.exit(-1)

	try:
		host_img = cv.LoadImage(sys.argv[1],cv.CV_LOAD_IMAGE_GRAYSCALE)
		feature_to_find = cv.LoadImage(sys.argv[2],cv.CV_LOAD_IMAGE_GRAYSCALE)
	except:
		print "******* Could not open image files *******"
		sys.exit(-1)

	img_size  = cv.GetSize(host_img)
	img_width = img_size[0]
	img_height = img_size[1]

	obj_size = cv.GetSize(feature_to_find) 
	obj_width = obj_size[0]
	obj_height = obj_size[1]
	print "image size =", img_size, " object to find size = ", obj_size

	cv.ShowImage("host_img", host_img)
	cv.ShowImage("feature_to_find", feature_to_find)
	cv.WaitKey()


	img1_copy = cv.CloneImage(host_img)
	img2_copy = cv.CloneImage(feature_to_find)
	cv.Smooth(img1_copy , img1_copy , cv.CV_GAUSSIAN,3, 3)
	cv.Smooth(img2_copy , img2_copy , cv.CV_GAUSSIAN, 3, 3)
	cv.Canny(img1_copy ,img1_copy  ,87,175, 3)
	cv.Canny(img2_copy ,img2_copy , 87,175, 3)

	cv.ShowImage("img1_copy ", img1_copy )
	cv.ShowImage("img2_copy ", img2_copy )
	cv.WaitKey()
	best_sum = 0
	best_orientation = (0,0)


for y in range (230, (img_height-obj_height), 1):
	for x in range(420,(img_width-obj_width),1):	
		#pt1 = [x,y]
		#pt2 = [x+obj_width, y+obj_height]
		#if (pt2[1] > img_height): pt2[1] = img_height
		#if (pt2[0] > img_width): pt2[0] = img_width
		#pt1 = tuple(pt1)
		#pt2 = tuple(pt2)
		#print pt1, pt2, pt2[0], img_width
		#cv.Rectangle(img, pt1, pt2, (1,1,1), thickness=1, lineType=8, shift=0)
		
		cropped_img = cv.GetSubRect(img1_copy, (x, y, obj_width, obj_height))
		#pp_cropped_img = cv.CreateImage( (obj_width, obj_height), 8, 1)
		#cv.Copy(src_region, cropped)
		#cv.Smooth(cropped_img , pp_cropped_img  , smoothtype=cv.CV_GAUSSIAN , param1=3, param2=3, param3=0, param4=0)
		#cv.Threshold(pp_cropped_img , pp_cropped_img , 128, 255, cv.CV_THRESH_BINARY)
		#cv.Canny(pp_cropped_img, pp_cropped_img  , 50 , 150)
		#cropped_hu =  cv.GetHuMoments(cv.Moments(pp_cropped_img)) 
		
		#try:
		#	storage = cv.CreateMemStorage() 
		#	(keypoints, descriptors) = cv.ExtractSURF(pp_cropped_img , None, storage , (1, 30, 3, 4))
		#except Exception, e:
        #		print e

		#print len(obj_keypoints), len(obj_descriptors), len(keypoints), len(descriptors)
		#print obj_keypoints[0]
		#for ((xx, yy), laplacian, size, dir, hessian) in keypoints:
		#	print "x=%d y=%d laplacian=%d size=%d dir=%f hessian=%f" % (xx, yy, laplacian, size, dir, hessian)
			#cv.Circle(cropped_img, (xx,yy), size, (255,0,0),1, cv.CV_AA , 0)
		
		#starpoints = cv.GetStarKeypoints(img, cv.CreateMemStorage(),)
		#print len (starpoints )
		#for ((xx, yy), size, response) in starpoints:
		#	print "x=%d y=%d size=%d response=%d" % (xx, yy, size, response)
		
		pil_img1 = Image.fromstring("L", cv.GetSize(cropped_img), cropped_img .tostring())
		pil_img2 = Image.fromstring("L", cv.GetSize(img2_copy), img2_copy.tostring())
		#s = 0
		#print pil_img1.getbands()
		#for band_index, band in enumerate(img1.getbands()):
	#		m1 = numpy.array([p[band_index] for p in img1.getdata()]).reshape(*img1.size)
	#		m2 = numpy.array([p[band_index] for p in img2.getdata()]).reshape(*img2.size)
	#		s += numpy.sum(numpy.abs(m1-m2))
		#h1 = numpy.array(img1.histogram())
		#h2 = numpy.array(img2.histogram())
		pixels1 = numpy.array(pil_img1)
		pixels2 = numpy.array(pil_img2)
		#a = numpy.array(cropped_hu)
		#b = numpy.array(obj_hu)
	    #obj_surf = numpy.array(flatten(obj_keypoints))
		#sample_surf = numpy.array(flatten(keypoints))
		#print b, sample_surf
		#rms = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b)**2, h1, h2))/len(h1))
		#rms2 = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b)**2, cropped_hu, obj_hu))/len(cropped_hu))
		pixels_dist = dist(pixels1, pixels2)
		#print pixels_dist
		#print "the root-mean-square (rms) dif =", rmsdiff(pil_img1, pil_img2)
		#print scipy.spatial.distance.euclidean(pil_img1, pil_img2)
		#print type(cropped_img), type(img2_copy)
		#get_orientation (cropped_img, img2_copy)
		subtracted_image = cv.CreateImage(cv.GetSize(img2_copy), 8, 1)
		cv.And(cropped_img, img2_copy , subtracted_image)
		cv.ShowImage("subtracted_image", subtracted_image)
		sum_of_and = cv.Sum(subtracted_image)
		if best_sum == 0: best_sum = sum_of_and[0]
		if sum_of_and[0] > best_sum: 
			best_sum = sum_of_and[0]
			best_orientation = (x,y)
			print "NEW HIGH Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "  best_orientation =", best_orientation	
			cv.WaitKey(0)
		#hu_dist = dist(b,a)
		#hist_dist = dist(h1,h2)
		#surf_dist = dist(sample_surf , obj_surf )
		#surf_dist = 0
		#if (hu_dist < low_hu): 
		#	low_hu = hu_dist
		#	print "NEW HU LOWS: ", low_hu
		#if (hist_dist < low_hist): 
		#	low_hist = hist_dist
		#	print "NEW HIST LOWS: ", low_hist
		#print "hist_dist= ", hist_dist, " hu_dist= ", hu_dist, " surf_dist = ", surf_dist, "   ",cropped_hu, "\n", obj_hu
		#t = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
		#tt = numpy.array([5.6])
		#print findclosest(tt, t)
		#cv.MatchTemplate(img, object_img, match, cv.CV_TM_SQDIFF)

		#for (result) in hh:
		#	print "result=%d " % result
		#cv.ShowImage("match", match )
		#cv.ShowImage("Object2",pp_obj_img)
		cv.ShowImage("cropped", cropped_img )
        #cv.ShowImage("canny", pp_cropped_img )
		#dist_euclidean = sqrt(sum((img - obj_img)^2)) / img_size
		#print dist_euclidean

		#dist_manhattan = sum(abs(i1 - i2)) / i1.size

		#dist_ncc = sum( (i1 - mean(i1)) * (i2 - mean(i2)) ) / ((i1.size - 1) * stdev(i1) * stdev(i2) )
		
		#cv.SetZero(cropped_img)  
		#print cv.GetSize(cropped_img)
		#cv.WaitKey(0)

#cv.ShowImage("Image",img )




cv.WaitKey(0)
