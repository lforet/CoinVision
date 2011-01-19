import sys, os
import cv

img = cv.LoadImage(sys.argv[1])
storage = cv.CreateMat(50, 1, cv.CV_32FC3)
cv.SetZero(storage)

temp = cv.CloneImage(img)
gray = cv.CreateImage(cv.GetSize(temp), 8, 1)	
edges = cv.CreateImage(cv.GetSize(temp), 8, 1)
cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

cv.Canny(gray, edges, 88, 175, 3)
cv.Smooth(edges, edges, cv.CV_GAUSSIAN, 9, 9)

circles = cv.HoughCircles(gray, storage, cv.CV_HOUGH_GRADIENT, 1, float(40), float(175), float(55), long(190),long(350) )
print storage
print cv.GetSize(storage)
num_of_circles = cv.GetSize(storage)[1]
print num_of_circles 
for i in range(num_of_circles):
	circle_data = storage[i,0]
	center = cv.Round(circle_data[0]), cv.Round(circle_data[1])
	radius = cv.Round(circle_data[2])
	print circle_data[0], circle_data[1], circle_data[2]




#for i in storage:
#	print [i,i]

cv.WaitKey()

