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
import ImageOps
from math import pi

def decimal2binary(n):
    '''convert denary integer n to binary string bStr'''
    bStr = ''
    if n < 0:  raise ValueError, "must be a positive integer"
    if n == 0: return '0'
    while n > 0:
        bStr = str(n % 2) + bStr
        n = n >> 1
    return bStr

def digitlist(value, numdigits=8, base=2):
	val = value
	return_str = ""
	digits = [0 for i in range(numdigits)]
	for i in range(numdigits):
		val, digits[i] = divmod(val, base)
		return_str = return_str + str(digits[i])
	#return_str = shift_string(return_str, numdigits-1)
	return digits

def shift_string(lst, n):
	first = (-n) % len(lst)
	result = list(lst[first:]) # if lst is a iterable but not a list
	result.extend(lst[:first])
	result = reduce(lambda x,y: x+y,result)
	return result

def get_uniform_variations(x):
	#this function gets all 8 variations of the bit pattern to establish the uniform pattern 
	# for example: 00000001 is the same rotational invariant LBP as 00000010 and 00000100  
	l = digitlist(x)
	l.reverse()
	#print "digit list = ", l	
	#print "reversed = ", l.reverse(), digitlist(x).reverse()
	l = reduce(lambda x,y: str(x) +str(y),l)
	#print 'l now = ', l
	#l= shift_string(l, len(l)-1)
	#print 'l now = ', l
	#print 'if',
	for c in range(len(l)):
		#print  'n ==', int(l, 2), 'or',
		#print  'l = ', int(l, 2) 
		l = shift_string(l, 1)
	#print ':histo[1] = histo[1] + 1'

def is_uniformLBP(digits):
	a = digits[0]
	transition_count = 0
	for i in range(1,len(digits)):
			if a != digits[i]:
				transition_count = transition_count + 1
				a = digits[i]
	if transition_count < 3:
		answer = True
	else: 
		answer = False
	return transition_count, answer


def image2array(im):
	#Create array of image using numpy
    return numpy.asarray(im)


def array2image(a):
#Create image from array
	return Image.fromarray(a)


def count_ones_in_string(digits):
		count = 0
		for n in digits:
			if n == 1: count = count + 1
		return count

def count_consecutive_ones(digits):
	consecutive_ones = 0
	highest_count = 0
	for i in digits:
			if i == 1: consecutive_ones  = consecutive_ones  + 1
			if consecutive_ones >= highest_count: highest_count = consecutive_ones 
			if i == 0: consecutive_ones  = 0
	print 'highest_count  = ', highest_count 
	return highest_count 

	
def get_LBP_uniform_histogram(lbp_list):
	# this function returns the histogram of the rotational invariant LBPs.  
	histo = [0] * 36
	#print "len of lbp_list = ", len(lbp_list)
	for n in lbp_list:
		#binary_string = digitlist(n)
		#transition_count = is_uniformLBP(binary_string)[0]
		#ones_count = count_ones_in_string(binary_string)
		#consecutive_ones = count_consecutive_ones(binary_string)
		#print n, binary_string , transition_count, ones_count, consecutive_ones
		#if (transition_count == 0) & (ones_count == 8):
		#	histo[0] = histo[0] + 1
		if n == 255:histo[0] = histo[0] + 1
		if n == 253 or n == 254 or n == 127 or n == 191 or n == 223 or n == 239 or n == 247 or n == 251 :histo[1] = histo[1] + 1
		if n == 249 or n == 252 or n == 126 or n == 63 or n == 159 or n == 207 or n == 231 or n == 243 :histo[2] = histo[2] + 1
		if n == 241 or n == 248 or n == 124 or n == 62 or n == 31 or n == 143 or n == 199 or n == 227 :histo[3] = histo[3] + 1
		if n == 225 or n == 240 or n == 120 or n == 60 or n == 30 or n == 15 or n == 135 or n == 195 :histo[4] = histo[4] + 1
		if n == 193 or n == 224 or n == 112 or n == 56 or n == 28 or n == 14 or n == 7 or n == 131 :histo[5] = histo[5] + 1
		if n == 129 or n == 192 or n == 96 or n == 48 or n == 24 or n == 12 or n == 6 or n == 3 :histo[6] = histo[6] + 1
		if n == 1 or n == 128 or n == 64 or n == 32 or n == 16 or n == 8 or n == 4 or n == 2 :histo[7] = histo[7] + 1
		if n == 0: histo[8] = histo[8] + 1
		if n == 245 or n == 250 or n == 125 or n == 190 or n == 95 or n == 175 or n == 215 or n == 235 :histo[9] = histo[9] + 1
		if n == 237 or n == 246 or n == 123 or n == 189 or n == 222 or n == 111 or n == 183 or n == 219 :histo[10] = histo[10] + 1
		if n == 221 or n == 238 or n == 119 or n == 187 or n == 221 or n == 238 or n == 119 or n == 187 :histo[11] = histo[11] + 1
		if n == 233 or n == 244 or n == 122 or n == 61 or n == 158 or n == 79 or n == 167 or n == 211 :histo[12] = histo[12] + 1
		if n == 229 or n == 242 or n == 121 or n == 188 or n == 94 or n == 47 or n == 151 or n == 203 :histo[13] = histo[13] + 1
		if n == 217 or n == 236 or n == 118 or n == 59 or n == 157 or n == 206 or n == 103 or n == 179 :histo[14] = histo[14] + 1
		if n == 213 or n == 234 or n == 117 or n == 186 or n == 93 or n == 174 or n == 87 or n == 171 :histo[15] = histo[15] + 1
		if n == 205 or n == 230 or n == 115 or n == 185 or n == 220 or n == 110 or n == 55 or n == 155 :histo[16] = histo[16] + 1
		if n == 181 or n == 218 or n == 109 or n == 182 or n == 91 or n == 173 or n == 214 or n == 107 :histo[17] = histo[17] + 1
		if n == 209 or n == 232 or n == 116 or n == 58 or n == 29 or n == 142 or n == 71 or n == 163  :histo[18] = histo[18] + 1
		if n == 201 or n == 228 or n == 114 or n == 57 or n == 156 or n == 78 or n == 39 or n == 147  :histo[19] = histo[19] + 1
		if n == 197 or n == 226 or n == 113 or n == 184 or n == 92 or n == 46 or n == 23 or n == 139  :histo[20] = histo[20] + 1
		if n == 177 or n == 216 or n == 108 or n == 54 or n == 27 or n == 141 or n == 198 or n == 99  :histo[21] = histo[21] + 1
		if n == 169 or n == 212 or n == 106 or n == 53 or n == 154 or n == 77 or n == 166 or n == 83  :histo[22] = histo[22] + 1
		if n == 165 or n == 210 or n == 105 or n == 180 or n == 90 or n == 45 or n == 150 or n == 75 :histo[23] = histo[23] + 1
		if n == 153 or n == 204 or n == 102 or n == 51 or n == 153 or n == 204 or n == 102 or n == 51 :histo[24] = histo[24] + 1
		if n == 149 or n == 202 or n == 101 or n == 178 or n == 89 or n == 172 or n == 86 or n == 43  :histo[25] = histo[25] + 1
		if n == 85 or n == 170 or n == 85 or n == 170 or n == 85 or n == 170 or n == 85 or n == 170  :histo[26] = histo[26] + 1
		if n == 161 or n == 208 or n == 104 or n == 52 or n == 26 or n == 13 or n == 134 or n == 67  :histo[27] = histo[27] + 1
		if n == 145 or n == 200 or n == 100 or n == 50 or n == 25 or n == 140 or n == 70 or n == 35  :histo[28] = histo[28] + 1
		if n == 137 or n == 196 or n == 98 or n == 49 or n == 152 or n == 76 or n == 38 or n == 19  :histo[29] = histo[29] + 1
		if n == 133 or n == 194 or n == 97 or n == 176 or n == 88 or n == 44 or n == 22 or n == 11  :histo[30] = histo[30] + 1
		if n == 81 or n == 168 or n == 84 or n == 42 or n == 21 or n == 138 or n == 69 or n == 162 :histo[31] = histo[31] + 1
		if n == 73 or n == 164 or n == 82 or n == 41 or n == 148 or n == 74 or n == 37 or n == 146 :histo[32] = histo[32] + 1
		if n == 65 or n == 160 or n == 80 or n == 40 or n == 20 or n == 10 or n == 5 or n == 130   :histo[33] = histo[33] + 1
		if n == 33 or n == 144 or n == 72 or n == 36 or n == 18 or n == 9 or n == 132 or n == 66   :histo[34] = histo[34] + 1
		if n == 17 or n == 136 or n == 68 or n == 34 or n == 17 or n == 136 or n == 68 or n == 34  :histo[35] = histo[35] + 1
	#print histo
	#print "sum of histo", sum(histo)
	return histo



def CalcLBP(img, radius=1, neighborPixels=8):
	#Function to calculate local binary pattern of an image 
	#pass in a img
	#returns an lbp list
	xmax = img.size[0]
	ymax = img.size[1] 
	#convert image to grayscale
	grayimage = ImageOps.grayscale(img)
	#make a copy to return
	returnimage = Image.new("L", (xmax,ymax))
	uniform_hist = numpy.zeros(256, int)
	#print "uniform_hist = ", uniform_hist
	#print "size = ", uniform_hist.ndim
	meanRGB = 0
	imagearray = grayimage.load()
	neighborRGB = numpy.empty([8], dtype=int)
	center = (5,5)
	radius = 1
	angleStep = 360 / neighborPixels
 	lbp_list = []
	for y in range(1, ymax-1, 1):
		for x in range(1, xmax-1, 1):
			centerRGB = imagearray[x, y]
			#print 'Center Pixel = ', x, y
			#meanRGB = centerRGB
			index = 0
			lbp = 0
			for ang in range(0, 359, angleStep):
				xx = round(x + radius * math.cos(math.radians(ang)))
				yy = round(y + radius * math.sin(math.radians(ang)))
				neighborRGB[index] = imagearray[xx,yy]
				#print 'pixel = ',xx, yy, 'neighbor# = ', index, 'center val = ', centerRGB, 'current pixel val = ', neighborRGB[index]
				if imagearray[xx,yy] >= centerRGB:
					lbp = lbp + (2**index)
				index = index + 1
			#print "LBP = ", lbp
			#lbp = 0
			#for i in range(neighborPixels):
			#comparing against mean adds a sec vs comparing against center pixel
				#if neighborRGB[i] >= meanRGB:
			#	if neighborRGB[i] >= centerRGB:
			#		lbp = lbp + (2**i)
			#comparing against mean adds a sec vs comparing against center pixel
			#meanRGB= centerRGB + neighborRGB.sum()
			#meanRGB = meanRGB / (neighbors+1)
			#compute Improved local binary pattern (center pixel vs the mean of neighbors)
			#putpixel adds 1 second vs storing to array
			uniform = is_uniformLBP( digitlist(lbp, numdigits=8, base=2))
			#print "LBP = ", lbp, " bits = ", decimal2binary(lbp), digitlist(lbp, numdigits=8, base=2), " is_uniformLBP(digits) = ", uniform
			get_uniform_variations(lbp)
			lbp_list.append(lbp)
			#time.sleep(1)
			returnimage.putpixel((x,y), lbp)
	#print "LBP LIST = ", lbp_list, "  count = ", len(lbp_list)
	LBP_Section_Histogram = get_LBP_uniform_histogram(lbp_list)
	#print 'LBP fingerprint for that section = ', LBP_Section_Histogram
	#print returnimage.histogram()
	#return returnimage
	return LBP_Section_Histogram

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
		#cv.ShowImage("subtracted_image", subtracted_image)
		#cv.ShowImage("Image of Interest", temp_img )
		sum_of_and = cv.Sum(subtracted_image)
		if best_sum == 0: best_sum = sum_of_and[0]
		if sum_of_and[0] > best_sum: 
			best_sum = sum_of_and[0]
			best_orientation = i
		#print i, "Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "  best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break
		time.sleep(.05)
	return (best_orientation)

def get_LBP_fingerprint(img, sections = 8):
	# - -------- this function takes and image and the number of sections to divide the image into (resolution of fingerprint)
	# ---------- returns a concatenated histogram will be the 'fingerprint' of the feature to find (the date) image
	img_size = img.size
	img_width = img_size[0]
	img_height = img_size[1]
	xsegs = img_width  / sections
	ysegs = img_height / sections
	fingerprint = []
	#print "xsegs, ysegs = ", xsegs, ysegs 
	#print obj_width % xsegs, obj_height % ysegs
	for yy in range(0,img_height-ysegs+1 , ysegs):
		for xx in range(0,img_width-xsegs+1,xsegs):
			#print "Processing section =", xx, yy, xx+xsegs, yy+ysegs
			pt1 = (xx, yy)
			pt2 = (xx+xsegs, yy+ysegs)
			box = (xx, yy, xx+xsegs, yy+ysegs)
			#print box
			cropped_img1 = img.crop(box)
			#cv.Rectangle(ftf_copy, pt1, pt2, cv.CV_RGB(255, 255, 255), 1, 0)
			#cropped_img1 = cv.GetSubRect(feature_to_find, (xx, yy,xsegs, ysegs))
			#pil_img1 = Image.fromstring("L", cv.GetSize(cropped_img1), cropped_img1.tostring())
			#temp  = CalcLBP(cropped_img1)
			fingerprint.extend(CalcLBP(cropped_img1))
			#cv.ShowImage("ftf_copy", ftf_copy)
			#cv.ShowImage("cropped_img1 ", cropped_img1 )
			#cv.WaitKey()
	#print 'THE ENTIRE FINGERPRINT = ', fingerprint
	return fingerprint



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
	ftf_copy = cv.CloneImage(feature_to_find)
	cv.Smooth(img1_copy , img1_copy , cv.CV_GAUSSIAN,3, 3)
	cv.Smooth(ftf_copy , ftf_copy, cv.CV_GAUSSIAN, 3, 3)
	cv.Canny(img1_copy ,img1_copy  ,87,175, 3)
	cv.Canny(ftf_copy,ftf_copy , 87,175, 3)

	#cv.ShowImage("img1_copy ", img1_copy )
	#cv.ShowImage("ftf_copy ", ftf_copy )
	#cv.WaitKey()
	best_sum = 0
	best_orientation = (0,0)
	best_hu = 0
	hu2 = numpy.array(cv.GetHuMoments(cv.Moments(feature_to_find)))
	

	#get fingerprint for feature to find (the date)
	pil_img1 = Image.fromstring("L", cv.GetSize(feature_to_find), feature_to_find.tostring())
	#pil_img1.show()
	#cv.WaitKey()
	lbp_h1 = numpy.array(get_LBP_fingerprint(pil_img1, sections = 4))
	#print type(h1), 'h1 = ', h1
	#cv.WaitKey()
	#pil_img1 = pil_img1.rotate(45)
	#pil_img1 = Image.fromstring("L", cv.GetSize(rotate_image(feature_to_find,1)), rotate_image(feature_to_find,1).tostring())
	#pil_img1.show()
	
	#h2 = numpy.array(get_LBP_fingerprint(pil_img1, sections = 1))
	#print dist(h1, h2)
	#print scipy.spatial.distance.euclidean(h1,h2)
	#cv.WaitKey()

for y in range (220, (img_height-(obj_height*2.5)), 5):
	for x in range(420,(img_width-(obj_width*2.5)), 5):	
		edge_cropped_img = cv.GetSubRect(img1_copy, (x, y, obj_width, obj_height))
		grey_cropped_img = cv.GetSubRect(host_img, (x, y, obj_width, obj_height))
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
		
		#pil_img1 = Image.fromstring("L", cv.GetSize(edge_cropped_img), cropped_img .tostring())
		#pil_img2 = Image.fromstring("L", cv.GetSize(img2_copy), img2_copy.tostring())
		#s = 0
		#print pil_img1.getbands()
		#for band_index, band in enumerate(img1.getbands()):
	#		m1 = numpy.array([p[band_index] for p in img1.getdata()]).reshape(*img1.size)
	#		m2 = numpy.array([p[band_index] for p in img2.getdata()]).reshape(*img2.size)
	#		s += numpy.sum(numpy.abs(m1-m2))
		#h1 = numpy.array(img1.histogram())
		#h2 = numpy.array(img2.histogram())
		#pixels1 = numpy.array(pil_img1)
		#pixels2 = numpy.array(pil_img2)
		#a = numpy.array(cropped_hu)
		#b = numpy.array(obj_hu)
	    #obj_surf = numpy.array(flatten(obj_keypoints))
		#sample_surf = numpy.array(flatten(keypoints))
		#print b, sample_surf
		#rms = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b)**2, h1, h2))/len(h1))
		#rms2 = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b)**2, cropped_hu, obj_hu))/len(cropped_hu))
		#pixels_dist = dist(pixels1, pixels2)
		
		#print pixels_dist
		#print "the root-mean-square (rms) dif =", rmsdiff(pil_img1, pil_img2)
		#print scipy.spatial.distance.euclidean(pil_img1, pil_img2)
		#print type(cropped_img), type(img2_copy)

		orientation = get_orientation (edge_cropped_img, ftf_copy)
		#rotate edge_cropped_img 

		subtracted_image = cv.CreateImage(cv.GetSize(ftf_copy), 8, 1)
		#hu1 = numpy.array(cv.GetHuMoments(cv.Moments(grey_cropped_img)))
		pil_img2 = Image.fromstring("L", cv.GetSize(grey_cropped_img), grey_cropped_img.tostring())
		lbp_h2 = numpy.array(get_LBP_fingerprint(pil_img2, sections = 4))
		hu_dist = dist(lbp_h1, lbp_h2)

		if best_hu == 0: best_hu = hu_dist
		if hu_dist < best_hu:
			best_hu = hu_dist
			cv.ShowImage("grey_cropped_img", grey_cropped_img)
			#cv.ShowImage("feature_to_find", feature_to_find)
			print "best_hu = ", best_hu
			cv.WaitKey(5)
		cv.ShowImage("sample section image", grey_cropped_img)
		cv.WaitKey(5)
#print 'done best hu = ', best_hu
		cv.And(edge_cropped_img, ftf_copy , subtracted_image)
		
		#cv.ShowImage("grey_cropped_img", grey_cropped_img)
		#cv.WaitKey()
		sum_of_and = cv.Sum(subtracted_image)
		if best_sum == 0: best_sum = sum_of_and[0]
		if sum_of_and[0] > best_sum: 
			best_sum = sum_of_and[0]
			best_orientation = (x,y)
			print "NEW HIGH Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "  best_orientation =", best_orientation
			cv.ShowImage("subtracted section image", grey_cropped_img)
			cv.WaitKey(5)	
			#cv.WaitKey()
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
		#cv.ShowImage("cropped", cropped_img )
        #cv.ShowImage("canny", pp_cropped_img )
		#dist_euclidean = sqrt(sum((img - obj_img)^2)) / img_size
		#print dist_euclidean

		#dist_manhattan = sum(abs(i1 - i2)) / i1.size

		#dist_ncc = sum( (i1 - mean(i1)) * (i2 - mean(i2)) ) / ((i1.size - 1) * stdev(i1) * stdev(i2) )
		
		#cv.SetZero(cropped_img)  
		#print cv.GetSize(cropped_img)
		#cv.WaitKey(0)
	
#cv.ShowImage("Image",img )




cv.WaitKey()
