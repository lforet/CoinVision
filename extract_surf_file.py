'''
Created on 21/04/2010

This script reads a set of image files (of the same pixels dimensions) 
from the directory defined in the PATH_ORIGIN variable . 
SURF descriptors are extracted for each of the read images. 

This script changes PATH_ORIGIN variable to the directory where your images are.
http://airobotics.ucn.cl/modules/vision
@author: jbekios
'''

import glob
from pygame.locals import *
import cv

cv.NamedWindow("SURF", 1)

PATH_ORIGIN = "../../../images/mark/*.jpg"
list_files = glob.glob(PATH_ORIGIN)
list_files.sort()

# Build a structure for storing image grayscale
im = cv.LoadImage(list_files[0], cv.CV_LOAD_IMAGE_COLOR)
imgray = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_8U, 1)

for file_image in list_files:
    im = cv.LoadImageM(file_image, cv.CV_LOAD_IMAGE_COLOR)
    cv.CvtColor(im, imgray, cv.CV_RGB2GRAY)
    try:
        (keypoints, descriptors) = cv.ExtractSURF(imgray, None, cv.CreateMemStorage(), (0, 3000, 3, 4))
        # DRAW KEYPOINT
        for ((x, y), laplacian, size, dir, hessian) in keypoints:
            radio = size*1.2/9.*2
            #print "radioOld: ", int(radio)
            color = (255, 0, 0)
            if radio < 3:
                radio = 2
                color = (0, 255, 0)
            #print "radioNew: ", int(radio)
            cv.Circle(im, (x,y), radio, (0,0,255))
            cv.ShowImage("SURF", im)
    except Exception, e:
        print e
    cv.WaitKey()
cv.DestroyAllWindows()    


