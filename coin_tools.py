"""Functions used in vision systems specifically for the recognition of coins """

import ImageChops
import math, operator
import sys
import cv
import cv2
import Image
import numpy
import numpy as np
import scipy.spatial
import time
from common import anorm
#from functools import partial
import mahotas
from scipy.misc import imread, imshow

def match_bruteforce(desc1, desc2, r_threshold = 0.75):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, n1))
    return np.array(res)

def match_flann(desc1, desc2, r_threshold = 0.6):
	FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
	flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
	flann = cv2.flann_Index(desc2, flann_params)
	idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
	mask = dist[:,0] / dist[:,1] < r_threshold
	idx1 = np.arange(len(desc1))
	pairs = np.int32( zip(idx1, idx2[:,0]) )
	return pairs[mask]

def draw_match(img1, img2, p1, p2, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))
    
    if status is None:
        status = np.ones(len(p1), np.bool_)
    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), status):
        col = [red, green][inlier]
        if inlier:
            cv2.line(vis, (x1, y1), (x2+w1, y2), col)
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2+w1, y2), 2, col, -1)
        else:
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2+w1-r, y2-r), (x2+w1+r, y2+r), col, thickness)
            cv2.line(vis, (x2+w1-r, y2+r), (x2+w1+r, y2-r), col, thickness)
    return vis


def match_and_draw(img1, img2, kp1, kp2, desc1, desc2, match, r_threshold):
    m = match(desc1, desc2, r_threshold)
    matched_p1 = np.array([kp1[i].pt for i, j in m])
    matched_p2 = np.array([kp2[j].pt for i, j in m])
    H, status = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 5.0)
    print '%d / %d  inliers/matched' % (np.sum(status), len(status))

    vis = draw_match(img1, img2, matched_p1, matched_p2, status, H)
    return vis


def resize_img(original_img, scale_percentage):
		print original_img.height, original_img.width, original_img.nChannels
		#resized_img = cv.CreateMat(original_img.rows * scale_percentage , original.cols * scale_percenta, cv.CV_8UC3)
		resized_img = cv.CreateImage((original_img.width * scale_percentage , original_img.height * scale_percentage), original_img.depth, original_img.nChannels)
		cv.Resize(original_img, resized_img)
		#cv.ShowImage("original_img", original_img)
		#cv.ShowImage("resized_img", resized_img)
		#cv.WaitKey()

def decimal2binary(n):
    """convert denary integer n to binary string bStr"""
    bStr = ''
    if n < 0:  raise ValueError, "must be a positive integer"
    if n == 0: return '0'
    while n > 0:
        bStr = str(n % 2) + bStr
        n = n >> 1
    return bStr


def digitlist(value, numdigits=8, base=2):
	"""returns representation of digits per params given: e.g. print digitlist(value=255, numdigits=8, base=2) returns [1,1,1,1,1,1,1,1] """ 
	val = value
	return_str = ""
	digits = [0 for i in range(numdigits)]
	for i in range(numdigits):
		val, digits[i] = divmod(val, base)
		return_str = return_str + str(digits[i])
	#return_str = shift_string(return_str, numdigits-1)
	digits.reverse()
	return digits

def image2array(img):
	"""given an image, returns an array. i.e. create array of image using numpy """
	return numpy.asarray(img)


def array2image(arry):
	"""given an array, returns an image. i.e. create image using numpy array """
	#Create image from array
	return Image.fromarray(arry)

def PILtoCV(PIL_img):
	cv_img = cv.CreateImageHeader(PIL_img.size, cv.IPL_DEPTH_8U, 1)
	cv.SetData(cv_img, img.tostring())
	return cv_img

def CVtoPIL(img):
	"""converts CV image to PIL image"""
	pil_img = Image.fromstring("L", cv.GetSize(img), img.tostring())
	cv_img = cv.CreateMatHeader(cv.GetSize(img)[1], cv.GetSize(img)[0], cv.CV_8UC1)
	cv.SetData(cv_img, pil_img.tostring())
	return pil_img

def rmsdiff(img1, img2):
    """Calculate the root-mean-square difference between two images"""

    h = ImageChops.difference(img1, img2).histogram()

    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(img1.size[0]) * img1.size[1]))
def cv2array(im):
  depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }

  arrdtype=im.depth
  a = np.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
  a.shape = (im.height,im.width,im.nChannels)
  return a

def array2cv(a):
  dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
          dtype2depth[str(a.dtype)],
          nChannels)
  cv.SetData(cv_im, a.tostring(),
             a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im
def get_LBP_fingerprint(img_cv, sections = 8):
	# - -------- this function takes and image and the number of sections to divide the image into (resolution of fingerprint)
	# ---------- returns a concatenated histogram will be the 'fingerprint' of the feature to find (the date) image
	img_size = cv.GetSize(img_cv)
	img_width = img_size[0]
	img_height = img_size[1]
	#print "imge size = img_wdith= ", img_width, "  img_height=", img_height, "  sections=", sections
	#cv.WaitKey()
	xsegs = img_width  / sections
	ysegs = img_height / sections
	fingerprint = []
	#print "xsegs, ysegs = ", xsegs, ysegs 
	#print obj_width % xsegs, obj_height % ysegs
	for yy in range(0, img_height-ysegs+1 , ysegs):
		for xx in range(0, img_width-xsegs+1, xsegs):
			#print "Processing section =", xx, yy, xx+xsegs, yy+ysegs
			#pt1 = (xx, yy)
			#pt2 = (xx+xsegs, yy+ysegs)
			box = (xx, yy, xsegs, ysegs)
			#print "box = ", box
			#cropped_img1 = img.crop(box)
			cropped_img1 = cv.GetSubRect(img_cv, box)
			cv.ShowImage("cropped_img1 ", cropped_img1 )
			#print "crop size", cv.GetSize(cropped_img1)
			cropped_img1 = cv.CloneMat(cropped_img1)
			cropped_img1 = cv.GetImage(cropped_img1)
			#cv.WaitKey()
			pixels = cv2array(cropped_img1)
			#pixels_avg = scipy.mean(pixels,2)
			lbp1 = mahotas.features.lbp(pixels , 1, 8, ignore_zeros=False)
			#print lbp1.ndim, lbp1.size
			#print "mahotas lbp histogram: ", lbp1
			fingerprint.append(lbp1)
	#print fingerprint.ndim, fingerprint.size
	#print 'THE ENTIRE FINGERPRINT = ', fingerprint[2], fingerprint[2][0]
	
	return fingerprint


