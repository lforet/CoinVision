import cv
import mahotas
import sys
from scipy.misc import imread, imshow
import scipy
from PIL import ImageOps
from PIL import Image
import numpy 
import time
import scipy.spatial
import ImageChops

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
	print return_str
	return digits

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


def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"

    h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))



def CalcLBP(img):
	#Function to calculate local binary pattern of an image 
	#pass in a img
	#returns an array???
	# Angle step.
	#PI = 3.14159265
	neighbors = 8
	#a = 2*PI/neighbors;
	#radius = 1
	#Increment = 1/neighbors
	xmax = img.size[0]
	ymax = img.size[1] 
	#convert image to grayscale
	grayimage = ImageOps.grayscale(img)
	#make a copy to return
	returnimage = Image.new("L", (xmax,ymax))
	neighborRGB = numpy.empty([8], dtype=int)
	meanRGB = 0
	imagearray = grayimage.load()
	for y in range(1, ymax-1, 1):				
		for x in range(1, xmax-1, 1):
			centerRGB = imagearray[x, y]
			meanRGB = centerRGB
			neighborRGB[4] = imagearray[x+1,y+1]
			neighborRGB[5] = imagearray[x,y+1]
			neighborRGB[6] = imagearray[x-1,y+1]
			neighborRGB[7] = imagearray[x-1,y]
			neighborRGB[0] = imagearray[x-1,y-1]
			neighborRGB[1] = imagearray[x,y-1]
			neighborRGB[2] = imagearray[x+1,y-1]
			neighborRGB[3] = imagearray[x+1,y]
			#comparing against mean adds a sec vs comparing against center pixel
			meanRGB= centerRGB + neighborRGB.sum()
			meanRGB = meanRGB / (neighbors+1)
			#compute Improved local binary pattern (center pixel vs the mean of neighbors)
			lbp = 0						
			for i in range(neighbors):
			#comparing against mean adds a sec vs comparing against center pixel
				if neighborRGB[i] >= meanRGB:
				#if neighborRGB[i] >= centerRGB:
					lbp = lbp + (2**i)
			#putpixel adds 1 second vs storing to array
			#print "lbp = ", lbp, " bits = ", decimal2binary(lbp), digitlist(lbp, numdigits=8, base=2), " is_uniformLBP(digits) = ", is_uniformLBP( digitlist(lbp, numdigits=8, base=2))
			#time.sleep(1)
			returnimage.putpixel((x,y), lbp)
	return returnimage

#def lbp(image, radius, points, ignore_zeros=False):

#img = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)

img  = imread(sys.argv[1])
#generates a RGB image, so do
#imshow(img)
print img.ndim
img2 = scipy.mean(img,2) # to get a 2-D array
#imshow(img2)

print img2.ndim
lbp1 = mahotas.features.lbp(img2, 1, 8, ignore_zeros=False)
print lbp1.ndim
print lbp1.size
print "mahotas lbp histogram: ", lbp1


img2 = array2image(img2)

for i in range (110,360,10):
	img4 = scipy.mean(img,2)
	img4 = array2image(img4)
	img4 = img4.rotate(i)
	img4 = image2array(img4)	
	img4 = array2image(img4)
	print "degrees: ", i, "  img2, img4:" ,scipy.spatial.distance.euclidean(img2, img4) 
	print rmsdiff(img2, img4)

lbp2 = mahotas.features.lbp(img4, 1, 8, ignore_zeros=False)
img2 = array2image(img2)
#imshow(img4)

#img3 = CalcLBP(img2)
#imshow(img3)

#h1 = numpy.array(img3.histogram())
#print h1

#img4 = CalcLBP(img4)
#img4.show()
#h2 = numpy.array(img4.histogram())
#print h2.size
#h2[255] = h1[255]
#print h2

print "lbp2: ", lbp2
print "img2, img4:" ,scipy.spatial.distance.euclidean(img2, img4) 


print lbp1.size, lbp1.shape, lbp1.ndim
print "Euclidian dist lpb1 , lbp2: ", scipy.spatial.distance.euclidean(lbp1, lbp2)
#print "euclidean:" ,scipy.spatial.distance.cdist( lbp1,  lbp2, 'euclidean')
print "minkowski:" ,scipy.spatial.distance.minkowski (lbp1,lbp2,2)
#print "seuclidean:" ,scipy.spatial.distance.seuclidean ( lbp1, lbp2, V= None )
print "correlation:", scipy.spatial.distance.correlation ( lbp1,lbp2)
print "braycurtis: ", scipy.spatial.distance.braycurtis (lbp1, lbp2)

#h1 = img2.histogram()
#img5 = scipy.mean(img,2)
#img5 = array2image(img5)
#img5 = img5.rotate(45)
#h2 = img5.histogram()
#img5.show()
#print "Euclidian dist h1 , h2: ", scipy.spatial.distance.euclidean(h1,h2)

#img1 = Image.fromstring("L", cv.GetSize(pp_obj_img), pp_obj_img.tostring())

