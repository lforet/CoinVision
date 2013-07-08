#!/usr/bin/python
# This is a standalone program. Pass an image name as a first parameter of the program.

import sys
from math import sin, cos, sqrt, pi
import cv
import urllib2
from coin_tools import *
import time
import scipy.spatial
# toggle between CV_HOUGH_STANDARD and CV_HOUGH_PROBILISTIC
USE_STANDARD = False
def houghlines(img, num_lines):
	"""
	Python: cv2.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines
	Parameters:	
	image - 8-bit, single-channel binary source image. The image may be modified by the function.
	lines - Output vector of lines. Each line is represented by a 4-element vector, where and are the ending points of each detected line segment.
	rho - Distance resolution of the accumulator in pixels.
	theta - Angle resolution of the accumulator in radians.
	threshold - Accumulator threshold parameter. Only those lines are returned that get enough votes (  ).
	minLineLength - Minimum line length. Line segments shorter than that are rejected.
	maxLineGap - Maximum allowed gap between points on the same line to link them.
	"""
	x = 400
	lines = np.array([[[]]])
	#print img
	while len(lines[0]) < num_lines:
		try:
			edges = cv2.Canny(img, (int(x/2)), x , apertureSize=3)
			#edges = img
			lines = cv2.HoughLinesP(edges, 1, math.pi/180, 50, None, 50, 10);
			if lines == None: 
				lines = np.array([[[]]])
			#print "x: ", x , " Lines: ", len(lines[0])
			x = x -5
		except:
			x = x -5
		#time.sleep(.2)
	#cv2.imwrite("canny.png", edges)
	print "Canny:", x
	#cv2.imwrite("cropped.png", temp_img)
	temp_top_lines = lines[0][:num_lines]
	top_lines = []
	#print temp_top_lines 
	
	for line in temp_top_lines:
		#print ([[line[0],line[1]]])
		dist = scipy.spatial.distance.cdist(([[line[0],line[1]]]), ([[line[2], line[3]]]), 'euclidean')
		#print "Line:", line, "  dist:", dist
		#top_lines.extend([line[0],line[1], line[2], line[3]])
		#top_lines.append(dist[0][0])
		top_lines.append([line[0],line[1], line[2], line[3], dist[0][0]])
	#print "houghlines:", top_lines, len(top_lines)
	top_lines = np.asarray(top_lines)
	top_lines = top_lines[top_lines[:,4].argsort()][::-1]#.flatten()
	print "sorted houghlines:", top_lines, len(top_lines)
	#top_lines = top_lines[:10]
	#print "###############################"
	#print top_lines
	###### Draw lines on temp img
	temp_img = img
	for line in top_lines:
		#print line, line[0]
		pt1 = (int(line[0]),int(line[1]))
		pt2 = (int(line[2]),int(line[3]))
		cv2.line(temp_img, pt1, pt2, (0,0,255), 2)
	cv2.imwrite("houghlines.png", temp_img)
	#time.sleep(2)

	x = np.array([top_lines[0][0], top_lines[0][1]])
	y = np.array([top_lines[0][2], top_lines[0][3]])
	z = np.array([top_lines[1][0], top_lines[1][1]])

	total_lines_point_dist = []

	for point in top_lines:

		line_pt1 =  np.array([point[0], point[1]])
		line_pt2 =  np.array([point[2], point[3]])
		pt1_x_dist = scipy.spatial.distance.cdist([x], [line_pt1], 'euclidean')
		pt1_y_dist = scipy.spatial.distance.cdist([y], [line_pt1], 'euclidean')
		pt1_z_dist = scipy.spatial.distance.cdist([z], [line_pt1], 'euclidean')
		pt2_x_dist = scipy.spatial.distance.cdist([x], [line_pt2], 'euclidean')
		pt2_y_dist = scipy.spatial.distance.cdist([y], [line_pt2], 'euclidean')
		pt2_z_dist = scipy.spatial.distance.cdist([z], [line_pt2], 'euclidean')
		#print line_pt1, pt1_x_dist, pt1_y_dist, pt1_z_dist, line_pt2, pt2_x_dist, pt2_y_dist, pt2_z_dist
		total_lines_point_dist.append([pt1_x_dist[0][0], pt1_y_dist[0][0], pt1_z_dist[0][0], pt2_x_dist[0][0], pt2_y_dist[0][0], pt2_z_dist[0][0]])
	total_lines_point_dist = np.array(total_lines_point_dist)#.flatten()
	
	print "total_lines_point_dist:"; print  total_lines_point_dist.flatten(); print
	
	#sys.exit(-1)
	print np.histogram(total_lines_point_dist, bins=10)
	return total_lines_point_dist.flatten()

	
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        src = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        url = 'https://code.ros.org/svn/opencv/trunk/opencv/doc/pics/building.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        src = cv.DecodeImageM(imagefiledata, cv.CV_LOAD_IMAGE_GRAYSCALE)


    cv.NamedWindow("Source", 1)
    cv.NamedWindow("Hough", 1)
    x = 500
    r = 0
    while True:
		if r > 360: r = 0
		dst = cv.CreateImage(cv.GetSize(src), 8, 1)
		rt = cv.CreateImage(cv.GetSize(src), 8, 1)
		rt = rotate_image(src, r)
		color_dst = cv.CreateImage(cv.GetSize(src), 8, 3)
		storage = cv.CreateMemStorage(0)
		lines = 0
		cv.Canny(rt, dst, x/2, x, 3)
		cv.CvtColor(dst, color_dst, cv.CV_GRAY2BGR)

		if USE_STANDARD:
			lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_STANDARD, 1, pi / 180, 100, 0, 0)
			for (rho, theta) in lines[:100]:
				a = cos(theta)
				b = sin(theta)
				x0 = a * rho 
				y0 = b * rho
				pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
				pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
				#if pt2[1] > 380 and pt2[1] < 550:
				cv.Line(color_dst, pt1, pt2, cv.RGB(255, 0, 0), 2, 8)
				#print rho, theta, pt1, pt2, a, b, x0, y0
		else:
			lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_PROBABILISTIC, 1, pi / 180, 50, 50, 10)
			print "Total lines:", len(lines)
			lines = lines[:25]
			for line in lines:
				cv.Line(color_dst, line[0], line[1], cv.CV_RGB(255, 0, 0), 2, 8)
			
		if r == 0:
			org_lines = lines
		if USE_STANDARD: print "STANDARD"
		else: print "Probalistic"
		print "canny:", x,  "  degrees:", r
		print "lines:", len(lines)
		#print "dist: ", scipy.spatial.distance.cdist(lines, org_lines,'euclidean')
		cv.ShowImage("Source", src)
		cv.ShowImage("Hough", color_dst)
		#if len(lines) < 125:
		#	k = ord("s")
		#	time.sleep(.2)
		#	cv.WaitKey(10)
		#if len(lines) > 125:
		#print "wait"
		k = cv.WaitKey(0)
		print k
		if k == ord(' ') or k == 1048608:
			USE_STANDARD = not USE_STANDARD
		if k == ord('w') or k == 1048695:
			x = x +10
		if k == ord('s') or k == 1048691:
			x = x  - 10	
		if k == ord('r') or k == 1048690:
			r = r + 5
		if k == ord('f') or k == 1048678:
			r = r - 5

		if k == ord('l') or k == 1048684:	
			if USE_STANDARD:
				for (rho, theta) in lines[:100]:
					a = cos(theta)
					b = sin(theta)
					x0 = a * rho 
					y0 = b * rho
					pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
					pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
					print rho, theta, pt1, pt2, a, b, x0, y0
			#print ([line[0], line[1]])
			else:
				for line in lines:
					print line, scipy.spatial.distance.cdist([line[0]], [line[1]],'euclidean')

		if k == 27 or k == 1048603:
			break

		if k == ord('h') or k == 1048680:
			houghlines(cv2array(rt), 25)	




