import ImageChops
import math, operator
import sys
import cv
import Image
import numpy
import scipy.spatial
import time


def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"

    h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))


img1 = cv.LoadImageM(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv.LoadImageM(sys.argv[2], cv.CV_LOAD_IMAGE_GRAYSCALE)


pil_img1 = Image.fromstring("L", cv.GetSize(img1), img1.tostring())
pil_img2 = Image.fromstring("L", cv.GetSize(img2), img2.tostring())

cv_img1 = cv.CreateMatHeader(cv.GetSize(img1)[1], cv.GetSize(img1)[0], cv.CV_8UC1)
temp_img = cv.CreateMatHeader(cv.GetSize(img1)[1], cv.GetSize(img1)[0], cv.CV_8UC1)
cv.SetData(cv_img1, pil_img1.tostring())
cv.SetData(temp_img, pil_img1.tostring())

cv.ShowImage("image", img1)
cv.ShowImage("image2", img2)
cv.WaitKey()


#rotate using opencv
start = time.time()
print start
mapMatrix = cv.CreateMat( 2, 3, cv.CV_32FC1 )
center = (cv.GetSize(img1)[0]/2, cv.GetSize(img1)[1]/2)
cv.GetRotationMatrix2D(center, 45, 1.0, mapMatrix)
cv.WarpAffine(cv_img1, temp_img, mapMatrix, flags=cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))
print "opencv time = " , (time.time() - start)
cv.ShowImage("Rotated_OPENCV", temp_img)
cv.WaitKey()

#rotate using PIL
start = time.time()
#pil_img1 = pil_img1.rotate(90, expand=True)
print pil_img1.size

pil_img1 = pil_img1.rotate(27)
#pil_img2 = pil_img1.rotate(46)
#pil_img3 = ImageChops.difference(pil_img2,pil_img1)
#pil_img3.show()

cv_im = cv.CreateImageHeader(pil_img1.size, cv.IPL_DEPTH_8U, 1)
cv.SetData(cv_im, pil_img1.tostring())
print "PIL time = ", (time.time() - start)
cv.WaitKey()
cv.ShowImage("Rotated_PIL", cv_im)
cv.WaitKey()

print type(img1), type(temp_img), type(cv_im)
print cv.GetSize(cv_img1), cv.GetSize(img1)


#pil_img.show()
#cv.WaitKey(0)
#
# rotate 60 degrees counter-clockwise
#
#pil_img1 = pil_img.rotate(1, expand=True)
#pil_img = pil_img.rotate(120)

print "the root-mean-square (rms) dif =", rmsdiff(pil_img1, pil_img2)
Y = scipy.spatial.distance.cdist(img1,img2, 'euclidean')
print pil_img1.size, Y.size, Y.shape, Y.ndim
print cv.Get2D(img1, 0, 0), cv.Get2D(img2, 0, 0), Y[0][0]
print scipy.spatial.distance.euclidean(cv.Get2D(img1, 0, 0),cv.Get2D(img2, 0, 0))
print scipy.spatial.distance.euclidean(img1[0][0],img2[0][0])
print scipy.spatial.distance.euclidean(img1[0],img2[0])
print scipy.spatial.distance.euclidean(img1,img2)
print scipy.spatial.distance.euclidean(img1, temp_img)
print scipy.spatial.distance.euclidean(img1, cv.GetMat(cv_im))
print scipy.spatial.distance.euclidean(temp_img, cv.GetMat(cv_im))

#Y = scipy.spatial.distance.cdist(img1,img2, 'correlation')
#print Y.size

#xx = numpy.array(shape=(1,1))

#xx = [(1,1,1,1), (2,2,2,2)]
#yy = [(1,1,1,1), (2,2,2,3)]

#print scipy.spatial.distance.euclidean(xx,yy)
