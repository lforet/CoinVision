import cv
import mahotas
import sys
from scipy.misc import imread, imshow
import scipy

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
print lbp1


#img1 = Image.fromstring("L", cv.GetSize(pp_obj_img), pp_obj_img.tostring())

