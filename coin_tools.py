"""Functions used in vision systems specifically for the recognition of coins """

import ImageChops
import math, operator
import sys
import cv
import Image
import numpy
import scipy.spatial
import time



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



