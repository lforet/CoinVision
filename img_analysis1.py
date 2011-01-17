from pylab import imshow, figure, zeros, plot
from scipy.misc import imread
from scipy.ndimage.interpolation import rotate
from numpy import savetxt
import sys
import cv 

a = imread(sys.argv[1],flatten=1)
imshow(a)

img_size = cv.GetSize(a)

center = [img_size[0]/2,img_size[1]/2]
 # define the center of the image (for cropping)
print center

width = 50 # choose a radius for the cropped image

crop = a[center[0]-width:center[0]+width,center[1]-width:center[1]+width]
imshow(crop)

stack = zeros((2000,10)) # create an array to save the slices
total = stack[:,0] # create an array to save the average slice

for i in range(10): # take ten slices
	stack[:,i] = rotate(crop,i*36,reshape=False)[50,:]
	total += stack[:,i]

plot(total) # plot the data
savetxt("Jefferson-v2.dat", total) # save the data to a file

