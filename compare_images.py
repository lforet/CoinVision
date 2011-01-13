import ImageChops
import math, operator
import sys
import cv
import Image
import numpy
import scipy.spatial


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
#pil_img.show()
#cv.WaitKey(0)
#
# rotate 60 degrees counter-clockwise
#
#pil_img = pil_img.rotate(120, expand=True)
#pil_img = pil_img.rotate(120)

print "the root-mean-square (rms) dif =", rmsdiff(pil_img1, pil_img2)
Y = scipy.spatial.distance.cdist(img1,img2, 'euclidean')
print pil_img1.size, Y.size, Y.shape, Y.ndim
print cv.Get2D(img1, 0, 0), cv.Get2D(img2, 0, 0), Y[0][0]
print scipy.spatial.distance.euclidean(cv.Get2D(img1, 0, 0),cv.Get2D(img2, 0, 0))
print scipy.spatial.distance.euclidean(img1[0][0],img2[0][0])
print scipy.spatial.distance.euclidean(img1[0],img2[0])
print scipy.spatial.distance.euclidean(img1,img2)

Y = scipy.spatial.distance.cdist(img1,img2, 'correlation')
print Y.size

xx = numpy.array(shape=(1,1))

xx = [(1,1,1,1), (2,2,2,2)]
yy = [(1,1,1,1), (2,2,2,3)]

print scipy.spatial.distance.euclidean(xx,yy)
