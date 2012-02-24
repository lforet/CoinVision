"""Functions used in vision systems specifically for the recognition of coins """

import ImageChops
import ImageEnhance
import ImageStat
import ImageFilter
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


visual = False




def CVtoGray(img):
	grey_image = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
	temp_img = cv.CloneImage(img)
	cv.CvtColor(temp_img, grey_image, cv.CV_RGB2GRAY)
	return grey_image

def grab_frame(camera):
	capture =  cv.CreateCameraCapture(camera)
	time.sleep(.1)
	frame = cv.QueryFrame(capture)
	time.sleep(.1)
	temp = cv.CloneImage(frame)
	return temp

def CV_enhance_edge(img):
	img1_copy = cv.CloneImage(img)
	img1_copy = CVtoPIL(img1_copy)
	#img1_copy = img1_copy.filter(ImageFilter.FIND_EDGES)
	img1_copy = img1_copy.filter(ImageFilter.EDGE_ENHANCE)
	img1_copy = PILtoCV(img1_copy)
	return img1_copy

###########################################################

def equalize_brightness(img1, img2): 
	#equalizes brightness of image to per image two. returns image #1 with brightness adjusted 
	img1_copy = cv.CloneImage(img1)
	img2_copy = cv.CloneImage(img2)

	img1_copy = CVtoPIL(img1_copy)
	img2_copy = CVtoPIL(img2_copy)

	#equalize brightness
	im_stat1 = ImageStat.Stat(img1_copy)
	img1_copy_mean = im_stat1.mean
	im_stat2 = ImageStat.Stat(img2_copy)
	img2_copy_mean = im_stat2.mean
	
	mean_ratio = 1 / (img1_copy_mean[0] / img2_copy_mean[0])
	
	if visual == True: 
			print "Normalizing Brightness of images..."
			print "img1_copy_mean:",img1_copy_mean,"  img2_copy_mean:", img2_copy_mean
			print "Adjusting img1 Brightness:", mean_ratio 
			print "Press any key to continue....."
			cv.WaitKey()
	enh = ImageEnhance.Brightness(img1_copy) 
	img1_copy = enh.enhance(mean_ratio)

	#img1_copy = img1_copy.filter(ImageFilter.FIND_EDGES)
	#img1_copy = img1_copy.filter(ImageFilter.EDGE_ENHANCE)
	#img1_copy = img1_copy.filter(ImageFilter.EMBOSS)
	#img1_copy = img1_copy.filter(ImageFilter.CONTOUR)
	#img1_copy = img1_copy.filter(ImageFilter.SMOOTH)
	#
	#img2_copy = img2_copy.filter(ImageFilter.FIND_EDGES)
	#img2_copy = img2_copy.filter(ImageFilter.EDGE_ENHANCE)
	#img2_copy = img2_copy.filter(ImageFilter.EMBOSS)
	#img2_copy = img2_copy.filter(ImageFilter.CONTOUR)
	#img2_copy = img2_copy.filter(ImageFilter.SMOOTH)

	img1_copy = PILtoCV(img1_copy)
	img2_copy = PILtoCV(img2_copy)
	return img1_copy

###########################################################

def surf_dif(img1, img2):
	#only features with a keypoint.hessian > 600 will be extracted
	#using extended descriptors (1) -> 128 elements each
	#surfParams = cvSURFParams(600, 1)
	#gray images for detecting
	object1 = cv.CreateImage((img1.width,img1.height), 8, 1)
	cv.CvtColor(img1, object1, cv.CV_BGR2GRAY)
	object2 = cv.CreateImage((img2.width,img2.height), 8, 1)
	cv.CvtColor(img2, object2, cv.CV_BGR2GRAY)

	keypoints1, descriptors1 = cv.ExtractSURF(object1, None, cv.CreateMemStorage(), (0, 400, 3, 4))
	keypoints2, descriptors2 = cv.ExtractSURF(object2, None, cv.CreateMemStorage(),(0, 400, 3, 4))

	print "found %d keypoints for img1"%len(keypoints1)
	print "found %d keypoints for img2"%len(keypoints2)

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


###########################################################

def match_bruteforce(desc1, desc2, r_threshold = 0.75):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, n1))
    return np.array(res)

###########################################################

def match_flann(desc1, desc2, r_threshold = 0.6):
	FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
	flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
	flann = cv2.flann_Index(desc2, flann_params)
	idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
	mask = dist[:,0] / dist[:,1] < r_threshold
	idx1 = np.arange(len(desc1))
	pairs = np.int32( zip(idx1, idx2[:,0]) )
	return pairs[mask]

###########################################################

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


###########################################################

def match_and_draw(img1, img2, kp1, kp2, desc1, desc2, match, r_threshold):
    m = match(desc1, desc2, r_threshold)
    matched_p1 = np.array([kp1[i].pt for i, j in m])
    matched_p2 = np.array([kp2[j].pt for i, j in m])
    H, status = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 5.0)
    print '%d / %d  inliers/matched' % (np.sum(status), len(status))

    vis = draw_match(img1, img2, matched_p1, matched_p2, status, H)
    return vis


###########################################################
	################# NOT FINISHED! 02/16/2012
def get_scaled_crops(img1, img2, sample_size):
	# finds center of two images
	# then scales first image up or down to match second
	#
	print "Finding center of coin 1....."
	coin1_center = find_center_of_coin(img1)
	print "center of coin 1.....", coin1_center
	#cv.Circle(img1, coin1_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
	#cv.ShowImage("Coin 1", img1)
	print "Finding center of coin 2....."
	coin2_center = find_center_of_coin(img2)
	print "center of coin 2.....", coin2_center
	#cv.Circle(img2, coin2_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
	#cv.ShowImage("Coin 2", img2)
	scaled_img = correct_scale(img1, img2, coin1_center, coin2_center)
	scaled_img_center = find_center_of_coin(scaled_img)
	#crop out center of coin based on found center
	print "Cropping center of original and scaled corrected images..."
	scaled_img_center_crop = center_crop(scaled_img, scaled_img_center, sample_size)
	coin2_center_crop = center_crop(img2, coin2_center, sample_size)
	coin1_center_crop = cv.CloneImage(scaled_img_center_crop)
	#cv.ShowImage("Crop Center of Coin1", coin1_center_crop)
	#cv.MoveWindow ('Crop Center of Coin1', (125 + (0 * cv.GetSize(coin1_center_crop)[1])) , (125 + (0 * cv.GetSize(coin1_center_crop)[0])) )
	#cv.ShowImage("Crop Center of Coin2", coin2_center_crop)
	#cv.MoveWindow ('Crop Center of Coin2', (125 + (1 * cv.GetSize(coin1_center_crop)[1])), (125 + (0 * cv.GetSize(coin2_center_crop)[0])) )
	#cv.WaitKey()
	return coin1_center_crop, coin2_center_crop


###########################################################

def center_crop(img, center, crop_size):
	#crop out center of coin based on found center
	x,y = center[0][0], center[0][1]
	#radius = center[1]
	radius = (crop_size * 4)
	tlx = x-(radius-crop_size)
	tly = y-(radius-crop_size)
	center_crop_topleft = (tlx, tly)
	brx = x+(radius-crop_size)
	bry = y+(radius-crop_size)
	center_crop_bottomright = (brx, bry)
	#print center_crop_bottomright 
	#print "crop top left:     ", center_crop_topleft
	#print "crop bottom right: ", center_crop_bottomright
	center_crop = cv.GetSubRect(img, (center_crop_topleft[0], center_crop_topleft[1] , (center_crop_bottomright[0] - center_crop_topleft[0]), (center_crop_bottomright[1] - center_crop_topleft[1])  ))
	center_crop_img = cv.CloneMat(center_crop)
	center_crop_img = cv.GetImage(center_crop_img)
	#if visual == True:
	#	print "center_crop_img:", center_crop_img 
	#	cv.ShowImage("Crop Center of Coin", center_crop)
	#	print "Press any key to continue....."	
	#	cv.WaitKey()
	#	cv.DestroyWindow("Crop Center of Coin")
	return center_crop_img

###########################################################

def resize_img(original_img, scale_percentage):
	#print original_img.height, original_img.width, original_img.nChannels
	#resized_img = cv.CreateMat(original_img.rows * scale_percentage , original.cols * scale_percenta, cv.CV_8UC3)
	resized_img = cv.CreateImage((cv.Round(original_img.width * scale_percentage) , cv.Round(original_img.height * scale_percentage)), original_img.depth, original_img.nChannels)
	cv.Resize(original_img, resized_img)
	if visual == True:
		print "resizing image..."
		cv.ShowImage("original_img", original_img)
		cv.ShowImage("resized_img", resized_img)
		print "Press any key to continue....."
		cv.WaitKey()
		cv.DestroyWindow("original_img")
		cv.DestroyWindow("resized_img")
	return(resized_img)
	
###########################################################

def decimal2binary(n):
    """convert denary integer n to binary string bStr"""
    bStr = ''
    if n < 0:  raise ValueError, "must be a positive integer"
    if n == 0: return '0'
    while n > 0:
        bStr = str(n % 2) + bStr
        n = n >> 1
    return bStr

###########################################################


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

###########################################################
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

###########################################################

def cv_img_distance(img1, img2, dist_type):

	img1_array = cv2array(img1)
	img1_array = img1_array.reshape(1, (img1_array.shape[0]*img1_array.shape[1]))
	img2_array = cv2array(img2)
	img2_array = img2_array.reshape(1, (img2_array.shape[0]*img2_array.shape[1]))

	if dist_type == 'euclidean':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'euclidean')

	if dist_type == 'seuclidean':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'seuclidean', V=None)

	if dist_type == 'cityblock':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'cityblock')

	if dist_type == 'cosine':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'cosine')

	if dist_type == 'correlation':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'correlation')

	if dist_type == 'hamming':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'hamming')

	if dist_type == 'chebyshev':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'chebyshev')

	if dist_type == 'braycurtis':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'braycurtis')

	if dist_type == 'mahalanobis':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'mahalanobis', VI=None)

	if dist_type == 'yule':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'yule')

	if dist_type == 'matching':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'matching')

	if dist_type == 'dice':
		distance_btw_images = scipy.spatial.distance.cdist(img1_array, img2_array,'dice')

	return distance_btw_images


def image2array(img):
	"""given an image, returns an array. i.e. create array of image using numpy """
	return numpy.asarray(img)

###########################################################

def array2image(arry):
	"""given an array, returns an image. i.e. create image using numpy array """
	#Create image from array
	return Image.fromarray(arry)

###########################################################

def PILtoCV(PIL_img):
	cv_img = cv.CreateImageHeader(PIL_img.size, cv.IPL_DEPTH_8U, 1)
	cv.SetData(cv_img, PIL_img.tostring())
	return cv_img

###########################################################

def CVtoPIL(img):
	"""converts CV image to PIL image"""
	cv_img = cv.CreateMatHeader(cv.GetSize(img)[1], cv.GetSize(img)[0], cv.CV_8UC1)
	#cv.SetData(cv_img, pil_img.tostring())
	pil_img = Image.fromstring("L", cv.GetSize(img), img.tostring())
	return pil_img
###########################################################

def rmsdiff(img1, img2):
    """Calculate the root-mean-square difference between two images"""

    h = ImageChops.difference(img1, img2).histogram()

    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(img1.size[0]) * img1.size[1]))

###########################################################


def rms_dist(x,y):   
    return numpy.sqrt(numpy.sum((x-y)**2))



############################################################

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

###########################################################

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


###########################################################

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



###########################################################

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
	#print "inside find_center_of_coin"
	#cv.WaitKey()
	best_circle = ((0,0),0)
	#minRadius = 100; maxRadius = 260
	canny = 150; param2 = 8;
	#for minRadius in range ((img.height/4), (img.height/2), 10):
	for minRadius in range (110, 190, 10):
		#img_copy = cv.CloneImage(img_copy2)
		#for maxRadius in range ((img.height/2)+50, img.height, 10):
		for maxRadius in range (190, 280, 10):
			#time.sleep(.01)
			#print "minRadius: ", minRadius, " maxRadius: ", maxRadius
			circles = cv.HoughCircles(edges, storage, cv.CV_HOUGH_GRADIENT, 1, img.height, canny, param2, minRadius, maxRadius)
	
			if storage.rows > 0:
				for i in range(0, storage.rows):
					#print "Center: X:", best_circle[0][0], " Y: ", best_circle[0][1], " Radius: ", best_circle[1], " minRadius: ", minRadius, " maxRadius: ", maxRadius
					cv.WaitKey(5)
					#time.sleep(.01)
					center = int(np.asarray(storage)[i][0][0]), int(np.asarray(storage)[i][0][1])
					radius = int(np.asarray(storage)[i][0][2])
					#print center, radius
					if visual == True:
						time.sleep(.05)
						cv.Circle(img_copy, center, radius, cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 )
						cv.Circle(img_copy, center, 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
						cv.ShowImage("Center of Coin", img_copy)
						cv.MoveWindow ('Center of Coin', 50 , (50 + (1 * (cv.GetSize(img_copy)[0]))))
					if (radius > best_circle[1]) & (radius > 150) & (radius < img.height/1.5):
						best_circle = (center, radius)
						if visual == True:
							print "Found Best Circle---Center: X:", best_circle[0][0], " Y: ", best_circle[0][1], " Radius: ", best_circle[1], " minRadius: ", minRadius, " maxRadius: ", maxRadius
	if visual == True:
		print "Finding centers completed..Press any key"
		cv.WaitKey()
		cv.DestroyWindow("Center of Coin")
	return best_circle

###########################################################

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
			cv.ShowImage("Fingerprint", cropped_img1 )
			cv.MoveWindow("Fingerprint", (101 + (2 * (cv.GetSize(cropped_img1)[0]))) , 100)
			#print "crop size", cv.GetSize(cropped_img1)
			cropped_img1 = cv.CloneMat(cropped_img1)
			cropped_img1 = cv.GetImage(cropped_img1)
			cv.WaitKey()
			pixels = cv2array(cropped_img1)
			#pixels_avg = scipy.mean(pixels,2)
			lbp1 = mahotas.features.lbp(pixels , 1, 8, ignore_zeros=False)
			#print lbp1.ndim, lbp1.size
			#print "mahotas lbp histogram: ", lbp1
			#cv.WaitKey()
			fingerprint.append(lbp1)
			#print "fingerprint=", fingerprint
			#fingerprint.extend([lbp1])
	#fingerprint = np.array(fingerprint)

	fingerprint = np.array(fingerprint)
	if visual == True:
		print "fingerprint.ndim, fingerprint.size=", fingerprint.ndim, fingerprint.size
		print 'THE ENTIRE FINGERPRINT = ', fingerprint	
		print "Press any key to contiue.."
		cv.WaitKey()
	cv.DestroyWindow("Fingerprint")
	#fingerprint = fingerprint.ravel()
	return fingerprint

###########################################################

def correct_scale(img1, img2, coin1_center, coin2_center):
	scaled_img = cv.CloneImage(img1)
	scale = float(coin2_center[1]) / float(coin1_center[1])
	print "Scaling image 1: ", scale,"%"
	scaled_img = resize_img(img1, scale)
	return(scaled_img)


###########################################################

def get_orientation_sobel(img1, img2, sample_size):
	print "Get Orientation Sobel"
	# rotate img1 the degrees returned by this function to match img2
	x = 120 #canny param
	sample_size = sample_size - 15
	# make copies
	img1_copy = cv.CloneImage(img1)
	img2_copy = cv.CloneImage(img2)

	#normalize brightness
	#make 1st image as bright as 2nd
	img1  = equalize_brightness(img1_copy , img2_copy )

	#perform filters
	#cv.Smooth(img2_copy , img2_copy, cv.CV_MEDIAN,3, 3)
	#cv.Canny(img2_copy , img2_copy  ,cv.Round((x/2)),x, 3)
	#cv.EqualizeHist(img2_copy, img2_copy)
	cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN,3, 3)
	#cv.Canny(img2_copy , img2_copy  ,cv.Round((x/2)),x, 3)
	sobel_img2_copy = cv.CreateImage(cv.GetSize(img2_copy), cv.IPL_DEPTH_16S,1)
	cv.Sobel(img2_copy, sobel_img2_copy, 1 , 1 )
	cv.ConvertScaleAbs(sobel_img2_copy, img2_copy, 1, 1)
	center = ([(img2_copy.width/2),(img2_copy.height/2)], sample_size)
	img2_copy =  center_crop(img2_copy, center, sample_size)
	subtracted_image = cv.CreateImage(cv.GetSize(img2_copy), 8, 1)
	cv.DestroyWindow("Orient Coin 2")
	cv.ShowImage  ("Orient Coin 2", img2_copy )
	cv.MoveWindow ('Orient Coin 2', (101 + (1 * (cv.GetSize(img2)[0]))) , (125 + (cv.GetSize(img2)[0])) )
	cv.WaitKey(5)
	best_sub = 9999999999
	#best_sub = 0
	best_orientation = 0
	print 'Starting to find best orientation'
	for i in range(0, 360, 1):
		img1_copy = rotate_image(img1, i)
		#cv.Smooth(img2_copy , img2_copy, cv.CV_MEDIAN,3, 3)
		#cv.Canny(img2_copy , img2_copy  ,cv.Round((x/2)),x, 3)
		cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
		#cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
		img1_copy =  center_crop(img1_copy, center, sample_size)
		sobel_img1_copy = cv.CreateImage(cv.GetSize(img1_copy), cv.IPL_DEPTH_16S,1)
		cv.Sobel(img1_copy, sobel_img1_copy, 1 , 1 )
		cv.ConvertScaleAbs(sobel_img1_copy, img1_copy, 1, 1)
		#the AND of two images has proven to be the most reliable 2/10/2012		
		cv.AbsDiff(img1_copy, img2_copy , subtracted_image)
		#cv.And(img1_copy, img2_copy , subtracted_image)
		#cv.Sub(img1_copy, img2_copy , subtracted_image)
		#cv.Max(img1_copy, img2_copy , subtracted_image)
		cv.ShowImage("Image 1 processed", img1_copy )
		cv.MoveWindow ("Image 1 processed", (100 + 2*cv.GetSize(img1_copy)[0]), 100)
		cv.ShowImage("Subtracted_Image", subtracted_image)
		cv.MoveWindow ("Subtracted_Image", (100 + 2*cv.GetSize(img1_copy)[0]), (150 + cv.GetSize(img1_copy)[1]) )
		result = cv.Sum(subtracted_image)	
		#print i, "result = ", result
		if result[0] < best_sub: 
			best_sub = result[0]
			best_orientation = i
			#print i, "result = ", result[0]
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break 
		#time.sleep(.01)
	print 'Finished finding best orientation'
	return (360-best_orientation)
	#return(best_sub)


###########################################################
#def compare_roi(img1, img2):
# this will compare three images (coins) and return 




###########################################################

def get_orientation_canny(img1, img2):
	#x=190 
	subtracted_image = cv.CreateImage(cv.GetSize(img1), 8, 1)
	temp_img = cv.CreateImage(cv.GetSize(img1), 8, 1)	
	best_sub = 999999999
	best_orientation = 0
	print 'Starting to find best orientation'
	best_canny  = 0
	best_dif = 9999999
	#for x in range(20, 200, 10):
	x = 160
	img1_copy = cv.CloneImage(img1)
	cv.Smooth(img1_copy , img1_copy, cv.CV_MEDIAN,3, 3)
	cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
	cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
	cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
	cv.ShowImage  ("Canny Coin 1", img1_copy )
	cv.MoveWindow ('Canny Coin 1', (101 + (1 * (cv.GetSize(img1)[0]))) , 100)
	for i in range(1, 360):
		img2_copy = cv.CloneImage(img2)
		img2_copy = rotate_image(img2_copy, i)
		cv.Smooth(img2_copy , img2_copy, cv.CV_MEDIAN,3, 3)
		cv.Canny(img2_copy , img2_copy  ,cv.Round((x/2)),x, 3)
		cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN,3, 3)
		cv.Canny(img2_copy , img2_copy  ,x/2, x, 3)
		cv.AbsDiff(img1_copy, img2_copy , subtracted_image)
		#cv.Sub(img1_copy, img2_copy , subtracted_image)
		cv.ShowImage  ("Canny Coin 2", img2_copy )
		cv.MoveWindow ('Canny Coin 2', (101 + (1 * (cv.GetSize(img1)[0]))) , (125 + (cv.GetSize(img1)[0])) )
		cv.ShowImage("Subtracted_Image", subtracted_image)
		cv.MoveWindow ("Subtracted_Image", (100 + 2*cv.GetSize(img1)[0]), (125 + cv.GetSize(img1)[1]) )
		result = cv.Sum(subtracted_image)	
		#print i, "result = ", result
		if result[0] < best_sub: 
			best_sub = result[0]
			best_orientation = i
			#print i, "result = ", result[0], "  best_orientation =", best_orientation
			#dif = math.fabs(265-best_orientation)
			#if dif < best_dif: 
			#	best_dif = dif
			#	best_canny = x
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break 
		#time.sleep(.01)
	print x, "   best canny: ", best_canny, "  best dif= ", best_dif
	print 'Finished finding best orientation'
	return (best_orientation)


###########################################################

def compare_images_rotation(img1, img2): 
	subtracted_image = cv.CreateImage(cv.GetSize(img1), 8, 1)
	img1_copy = cv.CloneImage(img1)
	temp_img = cv.CreateImage(cv.GetSize(img1), 8, 1)	
	cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
	sobel_img1_copy = cv.CreateImage(cv.GetSize(img1_copy), cv.IPL_DEPTH_16S,1)
	cv.Sobel(img1_copy, sobel_img1_copy, 1 , 1 )
	cv.ConvertScaleAbs(sobel_img1_copy, img1_copy, 1, 1)
	best_sub = 9999999999
	#best_sub = 0
	best_orientation = 0
	print 'Starting to find best orientation'
	for i in range(0, 360, 1):
		img2_copy = cv.CloneImage(img2)
		img2_copy = rotate_image(img2_copy, i)
		cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN,3, 3)
		sobel_img2_copy = cv.CreateImage(cv.GetSize(img2_copy), cv.IPL_DEPTH_16S,1)
		cv.Sobel(img2_copy, sobel_img2_copy, 1 , 1 )
		cv.ConvertScaleAbs(sobel_img2_copy, img2_copy, 1, 1)
		#the AND of two images has proven to be the most reliable 2/10/2012		
		#cv.AbsDiff(img1_copy, img2_copy , subtracted_image)
		#cv.And(img1_copy, img2_copy , subtracted_image)
		#cv.Sub(img1_copy, img2_copy , subtracted_image)
		#cv.Max(img1_copy, img2_copy , subtracted_image)
		cv.ShowImage("Image 2 being processed", img2_copy )
		cv.MoveWindow ("Image 2 being processed", (100 + 1*cv.GetSize(img2_copy)[0]), 100)
		#cv.ShowImage("Subtracted_Image", subtracted_image)
		#cv.MoveWindow ("Subtracted_Image", (100 + 1*cv.GetSize(img2_copy)[0]), (150 + cv.GetSize(img2_copy)[1]) )
		#result = cv.Sum(subtracted_image)	
		result = cv_img_distance(img1_copy, img2_copy, 'euclidean')[0]
		#print i, "result = ", result
		if result[0] < best_sub: 
			best_sub = result[0]
			best_orientation = i
			#print i, "result = ", result[0], "  best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break 
		#time.sleep(.01)
	#print 'Finished finding best orientation'
	#return (best_orientation)
	return(best_sub)

###########################################################

def compare_images_canny(img1, img2, sample_size):
		
	x = 120 #canny param
	sample_size = sample_size - 15
	# make copies
	img1_copy = cv.CloneImage(img1)
	img2_copy = cv.CloneImage(img2)

	#normalize brightness
	#make 1st image as bright as 2nd
	img1  = equalize_brightness(img1_copy , img2_copy )

	#perform filters
	#cv.Smooth(img2_copy , img2_copy, cv.CV_MEDIAN,3, 3)
	#cv.Canny(img2_copy , img2_copy  ,cv.Round((x/2)),x, 3)
	#cv.EqualizeHist(img2_copy, img2_copy)
	cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN,3, 3)
	#cv.Canny(img2_copy , img2_copy  ,cv.Round((x/2)),x, 3)
	sobel_img2_copy = cv.CreateImage(cv.GetSize(img2_copy), cv.IPL_DEPTH_16S,1)
	cv.Sobel(img2_copy, sobel_img2_copy, 1 , 1 )
	cv.ConvertScaleAbs(sobel_img2_copy, img2_copy, 1, 1)
	center = ([(img2_copy.width/2),(img2_copy.height/2)], sample_size)
	img2_copy =  center_crop(img2_copy, center, sample_size)
	subtracted_image = cv.CreateImage(cv.GetSize(img2_copy), 8, 1)
	cv.DestroyWindow("Canny Coin 2")
	cv.ShowImage  ("Canny Coin 2", img2_copy )
	cv.MoveWindow ('Canny Coin 2', (101 + (1 * (cv.GetSize(img2)[0]))) , (125 + (cv.GetSize(img2)[0])) )
	cv.WaitKey(5)
	best_sub = 9999999999
	#best_sub = 0
	best_orientation = 0
	print 'Starting to find best orientation'
	for i in range(0, 360, 1):
		img1_copy = rotate_image(img1, i)
		cv.Smooth(img1_copy , img1_copy, cv.CV_MEDIAN,3, 3)
		#cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
		cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
		#cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
		img1_copy =  center_crop(img1_copy, center, sample_size)
		sobel_img1_copy = cv.CreateImage(cv.GetSize(img1_copy), cv.IPL_DEPTH_16S,1)
		cv.Sobel(img1_copy, sobel_img1_copy, 1 , 1 )
		cv.ConvertScaleAbs(sobel_img1_copy, img1_copy, 1, 1)
		#the AND of two images has proven to be the most reliable 2/10/2012		
		#cv.AbsDiff(img1_copy, img2_copy , subtracted_image)
		cv.And(img1_copy, img2_copy , subtracted_image)
		#cv.Sub(img1_copy, img2_copy , subtracted_image)
		#cv.Max(img1_copy, img2_copy , subtracted_image)
		cv.ShowImage("Canny 1 processed", img1_copy )
		cv.MoveWindow ("Canny 1 processed", (100 + 2*cv.GetSize(img1_copy)[0]), 100)
		cv.ShowImage("Subtracted_Image", subtracted_image)
		cv.MoveWindow ("Subtracted_Image", (100 + 2*cv.GetSize(img1_copy)[0]), (150 + cv.GetSize(img1_copy)[1]) )
		result = cv.Sum(subtracted_image)	
		#print i, "result = ", result
		if result[0] > best_sub: 
			best_sub = result[0]
			best_orientation = i
			#print i, "result = ", result[0]
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break 
		#time.sleep(.01)
	#print 'Finished finding best orientation'
	#return (360-best_orientation)
	return(best_sub)




def compare_images_canny_sum(img1, img2):
	x = 180
	img1_copy = cv.CloneImage(img1)
	cv.Smooth(img1_copy , img1_copy, cv.CV_MEDIAN,3, 3)
	cv.EqualizeHist(img1_copy, img1_copy)
	cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
	#cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
	#cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
	cv.ShowImage  ("Canny Coin 1", img1_copy )
	cv.MoveWindow ('Canny Coin 1', (101 + (1 * (cv.GetSize(img1_copy)[0]))) , 100)
	img1_sum = cv.Sum(img1_copy)
	print (img1_sum)

	img2_copy = cv.CloneImage(img2)
	cv.Smooth(img2_copy , img2_copy, cv.CV_MEDIAN,3, 3)
	cv.EqualizeHist(img2_copy, img2_copy)
	cv.Canny(img2_copy , img2_copy  ,cv.Round((x/2)),x, 3)
	#cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN,3, 3)
	#cv.Canny(img2_copy , img2_copy  ,x/2, x, 3)
	cv.ShowImage  ("Canny Coin 2", img2_copy )
	cv.MoveWindow ('Canny Coin 2', (101 + (1 * (cv.GetSize(img2_copy)[0]))) , (125 + (cv.GetSize(img2_copy)[0])) )
	img2_sum = cv.Sum(img2_copy)
	print (img2_sum)

	result = math.fabs(img1_sum[0] - img2_sum[0])
	#hte less result the close the images are to each other (pixel count-wise)
	return (result)

#########################################################

def compare_images_lbp(img1, img2):
	x = 80
	img1 = equalize_brightness(img1, img2)
	
	img1_copy = cv.GetMat(img1)
	#cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
	#cv.EqualizeHist(img1_copy, img1_copy)
	#cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
	
	cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
	cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
	cv.DestroyWindow("LBP Coin 1")
	cv.ShowImage  ("LBP Coin 1", img1_copy )
	cv.MoveWindow ('LBP Coin 1', (101 + (1 * (cv.GetSize(img1_copy)[0]))) , 100)
	if visual == True:
		print "Smoothed and Canny Filter img1..."
		print "Press any key to continue..."
		cv.WaitKey()
	img1_lbp = get_LBP_fingerprint(img1_copy, sections = 1)
	img1_lbp = numpy.array(img1_lbp)
	img1_lbp = img1_lbp.reshape(1, (img1_lbp.shape[0]*img1_lbp.shape[1]))
	print "img1_lbp:", img1_lbp 
	cv.WaitKey(10)

	img2_copy = cv.GetMat(img2)
	#cv.Smooth(img2_copy , img2_copy, cv.CV_MEDIAN,3, 3)
	#cv.EqualizeHist(img2_copy, img2_copy)
	#cv.Canny(img2_copy , img2_copy  ,cv.Round((x/2)),x, 3)
	cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN,3, 3)
	cv.Canny(img2_copy , img2_copy  ,x/2, x, 3)
	cv.DestroyWindow("LBP Coin 2")
	cv.ShowImage  ("LBP Coin 2", img2_copy )
	cv.MoveWindow ('LBP Coin 2', (101 + (1 * (cv.GetSize(img2_copy)[0]))) , (125 + (cv.GetSize(img2_copy)[0])) )
	if visual == True:
		print "Smoothed and Canny Filter img2..."
		print "Press any key to continue..."
		cv.WaitKey()
	img2_lbp = get_LBP_fingerprint(img2_copy, sections = 1)
	img2_lbp = numpy.array(img2_lbp)
	img2_lbp = img2_lbp.reshape(1, (img2_lbp.shape[0]*img2_lbp.shape[1]))
	print "img2_lbp:", img2_lbp
	cv.WaitKey(10)
	distance = scipy.spatial.distance.cdist(img1_lbp, img2_lbp, 'euclidean')
	if visual == True:
		print "Completed LPB comparison.."
		print "Euclidian distance: ", distance
		print "Press any key to continue..."
		cv.WaitKey()
	return (distance)


###########################################################

def compare_images_laplace(img1, img2): 
	# rotate img2 the degrees returned by this function to make image match
	subtracted_image = cv.CreateImage(cv.GetSize(img1), 8, 1)
	img1_copy = cv.CloneImage(img1)
	temp_img = cv.CreateImage(cv.GetSize(img1), 8, 1)	
	#dst_16s2 = cv.CreateImage(cv.GetSize(img1), cv.IPL_DEPTH_16S, 1)
	#cv.Laplace(img1, dst_16s2,3)
	#cv.Convert(dst_16s2,img1_copy)
	#cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
	#sobel_img1_copy = cv.CreateImage(cv.GetSize(img1_copy), cv.IPL_DEPTH_16S,1)
	#cv.Sobel(img1_copy, sobel_img1_copy, 1 , 1 )
	#cv.ConvertScaleAbs(sobel_img1_copy, img1_copy, 1, 1)
	best_sub = 999999999999
	#best_sub = 0
	best_orientation = 0
	print 'Starting to find best orientation'
	for i in range(0, 360, 1):
		img2_copy = cv.CloneImage(img2)
		img2_copy = CVtoPIL(img2_copy)
		img2_copy = img2_copy.rotate(i, expand=1)
		img2_copy = PILtoCV(img2_copy)
		#img2_copy = rotate_image(img2_copy, i)
		dst_16s2 = cv.CreateImage(cv.GetSize(img2_copy), cv.IPL_DEPTH_16S, 1)
		cv.Laplace(img2_copy, dst_16s2,3)
		cv.Convert(dst_16s2,img2_copy)
		#cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN,3, 3)
		#sobel_img2_copy = cv.CreateImage(cv.GetSize(img2_copy), cv.IPL_DEPTH_16S,1)
		#cv.Sobel(img2_copy, sobel_img2_copy, 1 , 1 )
		#cv.ConvertScaleAbs(sobel_img2_copy, img2_copy, 1, 1)
		#the AND of two images has proven to be the most reliable 2/10/2012		
		cv.AbsDiff(img1_copy, img2_copy , subtracted_image)
		#cv.And(img1_copy, img2_copy , subtracted_image)
		#cv.Sub(img1_copy, img2_copy , subtracted_image)
		#cv.Max(img1_copy, img2_copy , subtracted_image)
		cv.ShowImage("Image 2 being processed", img2_copy )
		cv.MoveWindow ("Image 2 being processed", (100 + 1*cv.GetSize(img2_copy)[0]), 100)
		#cv.ShowImage("Subtracted_Image", subtracted_image)
		#cv.MoveWindow ("Subtracted_Image", (100 + 1*cv.GetSize(img2_copy)[0]), (150 + cv.GetSize(img2_copy)[1]) )
		result = cv.Sum(subtracted_image)
		#result = rmsdiff(img1_copy, img2_copy)	
		#print i, "result = ", result
		if result < best_sub: 
			best_sub = result
			best_orientation = i
			print i, "result = ", result, "  best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break 
		#time.sleep(.01)
	#print 'Finished finding best orientation'
	#return (best_orientation)
	return(best_sub)



###########################################################

def compare_images_brightness(img1, img2): 
	img1_copy = cv.CloneImage(img1)
	img2_copy = cv.CloneImage(img2)
	img1_copy = CVtoPIL(img1_copy)
	img2_copy = CVtoPIL(img2_copy)

	#equalize brightness
	im_stat1 = ImageStat.Stat(img1_copy)
	img1_copy_mean = im_stat1.mean
	im_stat2 = ImageStat.Stat(img2_copy)
	img2_copy_mean = im_stat2.mean
	#print "img1_copy_mean:",img1_copy_mean,"  img2_copy_mean:", img2_copy_mean
	mean_ratio = img1_copy_mean[0] / img2_copy_mean[0]
	#print "mean_ratio:", mean_ratio 
	#cv.WaitKey()
	enh = ImageEnhance.Brightness(img2_copy) 
	img2_copy = enh.enhance(mean_ratio)

	#
	#img1_copy = img1_copy.filter(ImageFilter.FIND_EDGES)
	#img1_copy = img1_copy.filter(ImageFilter.EDGE_ENHANCE)
	img1_copy = img1_copy.filter(ImageFilter.EMBOSS)
	#img1_copy = img1_copy.filter(ImageFilter.CONTOUR)
	img1_copy = img1_copy.filter(ImageFilter.SMOOTH)
	#
	#img2_copy = img2_copy.filter(ImageFilter.FIND_EDGES)
	#img2_copy = img2_copy.filter(ImageFilter.EDGE_ENHANCE)
	img2_copy = img2_copy.filter(ImageFilter.EMBOSS)
	#img2_copy = img2_copy.filter(ImageFilter.CONTOUR)
	img2_copy = img2_copy.filter(ImageFilter.SMOOTH)

	img1_copy = PILtoCV(img1_copy)
	img2_copy = PILtoCV(img2_copy)
	temp_copy = cv.CloneImage(img2_copy)

	cv.ShowImage("Image 1 being processed", img1_copy )
	cv.MoveWindow ("Image 1 being processed", (100 + 1*cv.GetSize(img1_copy)[0]), 100)
	best_sub = 9999999999
	#best_sub = 0
	best_orientation = 0
	print 'Starting to find best orientation'
	#for i in range(0, 360, 1):
	i = 0
	#	img2_copy = rotate_image(temp_copy, i)	
	cv.ShowImage("Image 2 being processed", img2_copy )
	cv.MoveWindow ("Image 2 being processed", (100 + 2*cv.GetSize(img2_copy)[0]), 100)
	#cv.WaitKey()
	result = cv_img_distance(img1_copy, img2_copy, 'euclidean')[0]
	#result = rmsdiff(img1_copy, img2_copy)	
	if result < best_sub: 
		best_sub = result
		best_orientation = i
		print i, "result = ", result, "  best_orientation =", best_orientation
	key = cv.WaitKey(50)
	#if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
	#	break 
	#time.sleep(.1)
	#print 'Finished finding best orientation'
	#return (best_orientation)
	return(best_sub)
	#distance = rmsdiff(img1_copy, img2_copy)
	#return distance


#######################################################

def compare_images_stddev(img1, img2): 
	img1_copy = CVtoPIL(img1)
	im_stat = ImageStat.Stat(img1_copy)
	img1_copy_stddev = im_stat.rms
	#best_sub = 9999999999
	best_sub = 0
	print 'Starting Standard Deviation Comparison'
	for i in range(0, 360, 360):
		img2_copy = CVtoPIL(img2)
		img2_copy = img2_copy.rotate(i)
		im_stat = ImageStat.Stat(img2_copy)
 		img2_copy_stddev = im_stat.rms
		#print "img1_copy_stddev:",img1_copy_stddev,"  img2_copy_stddev:", img2_copy_stddev
		#cv.WaitKey()
		stddev_ratio = img1_copy_stddev[0] / img2_copy_stddev[0]
		#print "stddev_ratio:", stddev_ratio 
		#cv.WaitKey()
		img2_copy = PILtoCV(img2_copy)
		cv.ShowImage("Image 2 being processed", img2_copy )
		cv.MoveWindow ("Image 2 being processed", (100 + 1*cv.GetSize(img2_copy)[0]), 100)
		#cv.WaitKey()
		img2_copy = CVtoPIL(img2_copy)
		result = stddev_ratio	
		if result > best_sub: 
			best_sub = result
			#print i, "result = ", result, "  best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break 
		#time.sleep(.01)
	return(best_sub)


#######################################################

def compare_images_var(img1, img2): 
	img1_copy = CVtoPIL(img1)
	im_stat = ImageStat.Stat(img1_copy)
	img1_copy_stddev = im_stat.var
	#best_sub = 9999999999
	best_sub = 0
	print 'Starting Variance Comparison'
	for i in range(0, 360, 1):
		img2_copy = CVtoPIL(img2)
		img2_copy = img2_copy.rotate(i)
		im_stat = ImageStat.Stat(img2_copy)
 		img2_copy_stddev = im_stat.var
		#print "img1_copy_stddev:",img1_copy_stddev,"  img2_copy_stddev:", img2_copy_stddev
		#cv.WaitKey()
		stddev_ratio = img1_copy_stddev[0] / img2_copy_stddev[0]
		#print "stddev_ratio:", stddev_ratio 
		#cv.WaitKey()
		img2_copy = PILtoCV(img2_copy)
		cv.ShowImage("Image 2 being processed", img2_copy )
		cv.MoveWindow ("Image 2 being processed", (100 + 1*cv.GetSize(img2_copy)[0]), 100)
		#cv.WaitKey()
		img2_copy = CVtoPIL(img2_copy)
		result = stddev_ratio	
		if result > best_sub: 
			best_sub = result
			#print i, "result = ", result, "  best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break 
		#time.sleep(.01)
	return(best_sub)



##############################################################


def compare_images_MatchTemp(img1, img2, sample_size): 
	print "Finding center of coin 1....."
	coin1_center = find_center_of_coin(img1)
	print "center of coin 1.....", coin1_center
	#cv.Circle(img1, coin1_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
	#cv.ShowImage("Coin 1", img1)
	print "Finding center of coin 2....."
	coin2_center = find_center_of_coin(img2)
	print "center of coin 2.....", coin2_center
	#cv.Circle(img2, coin2_center[0], 5, cv.CV_RGB(255, 0, 0), -1, cv.CV_AA, 0 )
	#cv.ShowImage("Coin 2", img2)
	#if scaled_img_center[1] == 0: 
	scaled_img = correct_scale(img1, img2, coin1_center, coin2_center)
	scaled_img_center = find_center_of_coin(scaled_img)
	#crop out center of coin based on found center
	print "Cropping center of original and scaled corrected images..."
	scaled_img_center_crop = center_crop(scaled_img, scaled_img_center, sample_size)
	cv.ShowImage("Crop Center of Scaled Coin1", scaled_img_center_crop)
	cv.MoveWindow ('Crop Center of Scaled Coin1', 100, 100)
	coin2_center = find_center_of_coin(img2)
	coin2_center_crop = center_crop(img2, coin2_center, sample_size)
	cv.ShowImage("Crop Center of Coin2", coin2_center_crop)
	cv.MoveWindow ('Crop Center of Coin2', 100, (125 + (cv.GetSize(coin2_center_crop)[0])) )
	
	#Create the result matrix
	match_cols =   coin2_center_crop.width  - scaled_img_center_crop.width + 1
	match_rows =   coin2_center_crop.height - scaled_img_center_crop.height + 1
	match_size = (match_cols, match_rows)
	#print match_size
	#print cv.GetSize(img1)

	match = cv.CreateMat( match_size[0], match_size[1] , cv.CV_32FC1)
	#object2 = cv.CreateImage((img2.width,img2.height), 8, 1)
	#coin2_center = find_center_of_coin(scaled_img_center_crop)
	#coin2_center_crop = center_crop(img2_copy, coin2_center, cv.Round(samples_size/4))
	#cv.ShowImage("Crop Center of Coin2", coin2_center_crop)
	#cv.MoveWindow ('Crop Center of Coin2', 100, (125 + (cv.GetSize(coin2_center_crop)[0])) )
	cv.MatchTemplate(coin2_center_crop, scaled_img_center_crop, match, cv.CV_TM_SQDIFF);
	#print match
	cv.ShowImage("match", match)

	cv.WaitKey()
	sys.exit(-1)

##############################################################
def compare_images_hu(img1, img2, sample_size):
	x = 120 #canny param
	sample_size = sample_size - 15
	# make copies
	img1_copy = cv.CloneImage(img1)
	img2_copy = cv.CloneImage(img2)

	#normalize brightness
	#make 1st image as bright as 2nd
	img1  = equalize_brightness(img1_copy , img2_copy )

	#perform filters
	#cv.Smooth(img2_copy , img2_copy, cv.CV_MEDIAN,3, 3)
	#cv.EqualizeHist(img2_copy, img2_copy)
	#cv.Canny(img2_copy , img2_copy  ,cv.Round((x/2)),x, 3)
	cv.Smooth(img2_copy , img2_copy, cv.CV_GAUSSIAN,3, 3)
	cv.Canny(img2_copy , img2_copy  ,x/2, x, 3)
	center = ([(img2_copy.width/2),(img2_copy.height/2)], sample_size)
	img2_copy =  center_crop(img2_copy, center, sample_size)

	cv.DestroyWindow("HU Coin 2")
	cv.ShowImage  ("HU Coin 2", img2_copy )
	cv.MoveWindow ('HU Coin 2', (101 + (1 * (cv.GetSize(img2)[0]))) , (125 + (cv.GetSize(img2)[0])) )
	#get hu moments
	img2_hu =  np.array(cv.GetHuMoments(cv.Moments(cv.GetMat(img2_copy))))	
	#reshape array to perform distance cals
	img2_hu = img2_hu.reshape(1, (img2_hu.shape[0]))
	cv.WaitKey(5)

	best_sub = 9999999999
	#best_sub = 0
	print 'Starting Hu Comparison'
	for i in range(0, 360, 1):
		img1_copy = rotate_image(img1, i)
		img1_copy =  center_crop(img1_copy, center, sample_size)
		#cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
		#cv.EqualizeHist(img1_copy, img1_copy)
		#cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
		cv.Smooth(img1_copy , img1_copy, cv.CV_GAUSSIAN,3, 3)
		cv.Canny(img1_copy , img1_copy  ,cv.Round((x/2)),x, 3)
		#cv.DestroyWindow("HU Coin 1")
		cv.ShowImage  ("HU Coin 1", img1_copy )
		cv.MoveWindow ('HU Coin 1', (101 + (1 * (cv.GetSize(img1)[0]))) , 100)
		cv.WaitKey(5)
		img1_hu =  np.array(cv.GetHuMoments(cv.Moments(cv.GetMat(img1_copy))))	
		img1_hu = img1_hu.reshape(1, (img1_hu.shape[0]))
		result = cv_img_distance(img1_copy, img2_copy, 'euclidean')[0]	
		if result< best_sub: 
			best_sub = result
			print i, "result = ", result
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break 
		#time.sleep(.01)
	return(best_sub)
	cv.WaitKey()
	return (distance)
##############################################################


def compare_images_orientation(img1, img2, sample_size):
		imgget_orientation_sobel(img1, img2, sample_size)


