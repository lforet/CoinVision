#!/usr/bin/env python

#This program will find and return the center of a coin image. It assumes only one coin per image

import cv
import optparse
import numpy
import Image 
import math, operator
import time
import scipy.spatial

#------Add option parameters	 
parser = optparse.OptionParser()

parser.add_option('-i', help='filename', dest='image',
    action='store_true')
parser.add_option('-o', help='object image', dest='objim',
    action='store_true')

(opts, args) = parser.parse_args()

# Making sure all mandatory options appeared.
#mandatories = ['image', 'pan']
mandatories = ['image', 'objim']
for m in mandatories:
    if not opts.__dict__[m]:
        print "mandatory option is missing\n"
        parser.print_help()
        exit(-1)

def dist(x,y):   
    return numpy.sqrt(numpy.sum((x-y)**2))

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
#
def findclosest(a,X):
	"""finds the vector in X most nearly parallel to a.
	This is the computationally expensive area, so optimize here."""
	if not len(X):
		return None
	current = X[0]
	smallest = numpy.inner(a-current, a-current)
	for item in X:
		d = numpy.inner(a-item, a-item)
		if d < smallest:
			current = item
			smallest = d
	return current


def findclosest_lpf(a,X):
	"""finds the vector in X most nearly parallel to a.
	This is the computationally expensive area, so optimize here."""
	if not len(X):
		return None
	current = X[0]
	smallest = scipy.spatial.distance.euclidean(a,X)
	for item in X:
		d = numpy.inner(a-item, a-item)
		if d < smallest:
			current = item
			smallest = d
	return current

def findclosesttaxicab(a,X):
    """finds the vector in X most nearly parallel to a using taxicab metric."""

    if not len(X):
        return None
    current = X[0]
    smallest = numpy.abs(a-current).sum()
    for item in X:
        d = numpy.abs(a-item).sum()
        if d < smallest:
            current = item
            smallest = d
    return current

def find_first_n(data,X, n=3):
    """
    find_first_n(data, X, n=3) --> [(fit,vector, match), ...]

    Finds up to first n vectors along with their matches, sorted by
    order of fit."""

    if len(data) <= n:
        return data
    else:
        # A little ugly, but we can't use dictionaries, since vectors may not
        # be unique AND because dictionaries don't have ordering.

        # create a list of three winners: [(fit,vector,match),...] ordered
        # by fit from smallest to largest.
        winners = []
        for i in range(n):
            match = findclosest(data[i],X)
            fit = numpy.inner(data[i],match)
            for x in range(len(winners)):
                if fit < winners[x][0]:
                    winners = (winners[:x] + [(fit,data[i],match)] + winners[x:])
                    break
            else:
                winners.append((fit,data[i],match))
                
        return winners

def find_best_n(data, X, n=3):
    """
    find_best_three(data, X, n=3) --> [(fit,vector, match), ...]

    Finds best n vectors along with their matches, sorted by
    order of fit."""

    if len(data) <= n:
        return data
    # Prime the pump
    winners = find_first_n(data,X,n)

    # We've already examined the first n...
    for i in range(n,len(data)):
        match = findclosest(data[i],X)
        fit = numpy.inner(data[i],match)
        for x in range(n):
            if fit < winners[x][0]:
                winners = (winners[:x] + [(fit,data[i],match)] + winners[x:])[:n]
                break
    return winners

X = numpy.random.rand(60000,30)
#ll = numpy.random.rand(10,1,1)
print X[0]
print X[0].ndim

data = numpy.random.rand(200,30)
print X.shape
print X.ndim
print X.size

#X.reshape(222000,1)
print X.ndim

# optimization metrics
start = time.time()
for i in data:
	print i.ndim, X.ndim
	#findclosest_lpf(i,X)
	findclosest(i,X)
stop = time.time()
print "findclosest: Dt = %0.2f for %d matches in array of size %d" %\
      (stop - start,len(data),len(X))
"""
start = time.time()
for i in data:
    findclosesttaxicab(i,X)
stop = time.time()
print "findclosesttaxicab: Dt = %0.2f for %d matches in array of size %d" %\
      (stop - start,len(data),len(X))
# END optimization metrics
"""
#load image
img = cv.LoadImageM(args[0], cv.CV_LOAD_IMAGE_GRAYSCALE)
object_img = cv.LoadImageM(args[1], cv.CV_LOAD_IMAGE_GRAYSCALE)

img_size  = cv.GetSize(img)
img_width = img_size[0]
img_height = img_size[1]

obj_size = cv.GetSize(object_img) 
obj_width = obj_size[0]
obj_height = obj_size[1]
print "image size =", img_size, " object to find size = ", obj_size

#sobel_img = cv.CreateImage( (img_width, img_height), 32, 1)
#cv.Sobel(img, sobel_img,0, 1, apertureSize = 1)
#cv.ShowImage("Sobel",sobel_img )
#canny_img = cv.CreateImage( (img_width, img_height), 8, 1)
#cv.Canny(img, canny_img,50,150)
#cv.ShowImage("canny",canny_img )
#cv.WaitKey(0)


pp_obj_img = cv.CreateImage( (obj_width, obj_height), 8, 1)
match = cv.CreateImage( ((img_width-obj_width+1), (img_height-obj_height+1)), 32, 1)

cv.Smooth(object_img, pp_obj_img , smoothtype=cv.CV_GAUSSIAN , param1=3, param2=3, param3=0, param4=0)
#cv.ShowImage("Object2",pp_obj_img)
#cv.WaitKey(0)
#cv.Threshold(pp_obj_img , pp_obj_img , 138, 255, cv.CV_THRESH_BINARY)
#cv.ShowImage("Object2",pp_obj_img)
#cv.WaitKey(0)
#cv.Dilate(pp_obj_img , pp_obj_img , iterations=1)
#cv.ShowImage("Object2",pp_obj_img)
#cv.WaitKey(0)
#cv.Erode(pp_obj_img , pp_obj_img , iterations=1)
#cv.ShowImage("Object2",pp_obj_img)
#cv.WaitKey(0)
cv.Canny(pp_obj_img,pp_obj_img , 50 , 150)
#cv.ShowImage("Object2",pp_obj_img)
#cv.WaitKey(0)

obj_hu =  cv.GetHuMoments(cv.Moments(pp_obj_img)) 

for a in range (0, 360, 36):
	img1 = Image.fromstring("L", cv.GetSize(pp_obj_img), object_img.tostring())
	img1 = img1.rotate(a, expand=True)	
	cv_im = cv.CreateImageHeader(img1.size, cv.IPL_DEPTH_8U, 1)
	cv.SetData(cv_im, img1.tostring())
	hu1 = cv.GetHuMoments(cv.Moments(cv_im)) 	
 	print "angle = ", a , " hu =  ", hu1

try:
	storage = cv.CreateMemStorage() 
	(obj_keypoints, obj_descriptors) = cv.ExtractSURF(pp_obj_img, None, storage , (1, 30, 3, 4))
	print len(obj_keypoints), len(obj_descriptors)
except Exception, e:
	print e

#for ((x, y), laplacian, size, dir, hessian) in keypoints:
#	print "x=%d y=%d laplacian=%d size=%d dir=%f hessian=%f" % (x, y, laplacian, size, dir, hessian)
	#cv.Circle(object_img , (x,y), size, (255,0,0),1, cv.CV_AA , 0)



cropped = cv.CreateImage( (obj_width, obj_height), 8, 1)

low_hist = 1000
low_hu = 1000
cv.ShowImage("Object2",pp_obj_img)
cv.ShowImage("Image",img )
cv.ShowImage("Object",object_img )
cv.WaitKey(0)


for y in range (230, (img_height-obj_height), 5):
	for x in range(420,(img_width-obj_width),5):	
		#pt1 = [x,y]
		#pt2 = [x+obj_width, y+obj_height]
		#if (pt2[1] > img_height): pt2[1] = img_height
		#if (pt2[0] > img_width): pt2[0] = img_width
		#pt1 = tuple(pt1)
		#pt2 = tuple(pt2)
		#print pt1, pt2, pt2[0], img_width
		#cv.Rectangle(img, pt1, pt2, (1,1,1), thickness=1, lineType=8, shift=0)
		
		cropped_img = cv.GetSubRect(img, (x, y, obj_width, obj_height))
		pp_cropped_img = cv.CreateImage( (obj_width, obj_height), 8, 1)
		cv.Smooth(cropped_img , pp_cropped_img  , smoothtype=cv.CV_GAUSSIAN , param1=3, param2=3, param3=0, param4=0)
		#cv.Threshold(pp_cropped_img , pp_cropped_img , 128, 255, cv.CV_THRESH_BINARY)
		cv.Canny(pp_cropped_img, pp_cropped_img  , 50 , 150)
		cropped_hu =  cv.GetHuMoments(cv.Moments(pp_cropped_img)) 
		#cv.Copy(src_region, cropped)
		try:
			storage = cv.CreateMemStorage() 
			(keypoints, descriptors) = cv.ExtractSURF(pp_cropped_img , None, storage , (1, 30, 3, 4))
		except Exception, e:
        		print e

		print len(obj_keypoints), len(obj_descriptors), len(keypoints), len(descriptors)
		#print obj_keypoints[0]
		#for ((xx, yy), laplacian, size, dir, hessian) in keypoints:
		#	print "x=%d y=%d laplacian=%d size=%d dir=%f hessian=%f" % (xx, yy, laplacian, size, dir, hessian)
			#cv.Circle(cropped_img, (xx,yy), size, (255,0,0),1, cv.CV_AA , 0)
		
		#starpoints = cv.GetStarKeypoints(img, cv.CreateMemStorage(),)
		#print len (starpoints )
		#for ((xx, yy), size, response) in starpoints:
		#	print "x=%d y=%d size=%d response=%d" % (xx, yy, size, response)
		
		img1 = Image.fromstring("L", cv.GetSize(pp_obj_img), pp_obj_img.tostring())
		img2 = Image.fromstring("L", cv.GetSize(pp_cropped_img), pp_cropped_img.tostring())
		s = 0
		print img1.getbands()
		#for band_index, band in enumerate(img1.getbands()):
	#		m1 = numpy.array([p[band_index] for p in img1.getdata()]).reshape(*img1.size)
	#		m2 = numpy.array([p[band_index] for p in img2.getdata()]).reshape(*img2.size)
	#		s += numpy.sum(numpy.abs(m1-m2))
		h1 = numpy.array(img1.histogram())
		h2 = numpy.array(img2.histogram())
		a = numpy.array(cropped_hu)
		b = numpy.array(obj_hu)
		obj_surf = numpy.array(flatten(obj_keypoints))
		sample_surf = numpy.array(flatten(keypoints))
		print b, sample_surf
		#rms = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b)**2, h1, h2))/len(h1))
		#rms2 = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b)**2, cropped_hu, obj_hu))/len(cropped_hu))
		hu_dist = dist(b,a)
		hist_dist = dist(h1,h2)
		#surf_dist = dist(sample_surf , obj_surf )
		surf_dist = 0
		if (hu_dist < low_hu): 
			low_hu = hu_dist
			print "NEW HU LOWS: ", low_hu
		if (hist_dist < low_hist): 
			low_hist = hist_dist
			print "NEW HIST LOWS: ", low_hist
		print "hist_dist= ", hist_dist, " hu_dist= ", hu_dist, " surf_dist = ", surf_dist, "   ",cropped_hu, "\n", obj_hu
		t = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
		tt = numpy.array([5.6])
		print findclosest(tt, t)
		#cv.MatchTemplate(img, object_img, match, cv.CV_TM_SQDIFF)

		#for (result) in hh:
		#	print "result=%d " % result
		#cv.ShowImage("match", match )
		cv.ShowImage("Object2",pp_obj_img)
		cv.ShowImage("cropped", cropped_img )
                cv.ShowImage("canny", pp_cropped_img )
		#dist_euclidean = sqrt(sum((img - obj_img)^2)) / img_size
		#print dist_euclidean

		#dist_manhattan = sum(abs(i1 - i2)) / i1.size

		#dist_ncc = sum( (i1 - mean(i1)) * (i2 - mean(i2)) ) / ((i1.size - 1) * stdev(i1) * stdev(i2) )
		
		#cv.SetZero(cropped_img)  
		#print cv.GetSize(cropped_img)
		cv.WaitKey(0)

cv.ShowImage("Image",img )




cv.WaitKey(0)
