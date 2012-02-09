



import cv
#from opencv import cv2
from SimpleCV import *
import sys
import numpy as np
import Image 
from coin_tools import *


if __name__=="__main__":

	if len(sys.argv) < 2:
		#print "******* Requires an image files of the same size."
		print "This program will return the angle at which the second is in relation to the first. ***"
		sys.exit(-1)

	try:
		img1 = cv.LoadImage(sys.argv[1],cv.CV_LOAD_IMAGE_GRAYSCALE)
		img2 = cv.LoadImage(sys.argv[2],cv.CV_LOAD_IMAGE_GRAYSCALE)
	except:
		print "******* Could not open image files *******"
		sys.exit(-1)

	cv.ShowImage("Image 1", img1)
	cv.MoveWindow ('Image 1',50 ,50 )
	cv.ShowImage("Image 2", img2)
	cv.MoveWindow ('Image 2', (50 + (1 * (cv.GetSize(img1)[0]))) , 50)
	cv.WaitKey()

	fp1 = get_LBP_fingerprint(img1, 4)
	fp2 = get_LBP_fingerprint(img2, 4)

	print "Euclidian dist h1 , h2: ", scipy.spatial.distance.euclidean(fp1,fp2)
