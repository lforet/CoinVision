import sys, os
import cv
import Image

def Subtract_Images(img1, img2):
	gray1 = cv.CreateImage(cv.GetSize(img1), 8, 1)
	gray2 = cv.CreateImage(cv.GetSize(img2), 8, 1)
	cv.CvtColor(img1, gray1, cv.CV_BGR2GRAY)
	#cv.CvtColor(img2, gray2, cv.CV_BGR2GRAY)
	cv.Smooth(gray1, gray1, cv.CV_GAUSSIAN, 9, 9)
	cv.Smooth(gray2, gray2, cv.CV_GAUSSIAN, 9, 9)
	cv.Canny(gray1,gray1 ,87,175, 3)
	cv.Canny(gray2,gray2, 87,175, 3)
	cv.ShowImage("gray1", gray1)
	cv.ShowImage("gray2", gray2)

def Draw_Boundries(img, coin_center):
	print coin_center
	size_buffer = 15
	radius_buffer = 50
	center = int(coin_center[0]), int(coin_center[1])
	radius = int(cv.Round(coin_center[2]))
	inside_radius = radius - radius_buffer

	temp = cv.CloneImage(img)
	cv.Circle(temp, (center), radius, cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 )
	cv.Circle(temp ,(center), 2, cv.CV_RGB(0, 0, 255), 2, cv.CV_AA, 0 ) 	
	cv.Circle(temp ,(center), (radius - radius_buffer), cv.CV_RGB(0, 0, 255), 2, cv.CV_AA, 0 )

	#Draw outside bounding rectangle 
	topleft_corner = (center[0]-radius-size_buffer, center[1]-radius-size_buffer)
	bottomright_corner = (center[0]+radius+size_buffer, center[1]+radius+size_buffer)
	cv.Rectangle(temp, topleft_corner, bottomright_corner, cv.CV_RGB(255, 255, 0), 2, 0)
	#Draw inside bounding rectangle 
	print inside_radius
	print (inside_radius*(cv.Sqrt(2)/2))
	topleft_corner = (center[0]-int((inside_radius*(cv.Sqrt(2)/2))), center[1]-int((inside_radius*(cv.Sqrt(2)/2))))
	bottomright_corner = (center[0]+int((inside_radius*(cv.Sqrt(2)/2))), center[1]+int((inside_radius*(cv.Sqrt(2)/2))))
	cv.Rectangle(temp, topleft_corner, bottomright_corner, cv.CV_RGB(255, 255, 0), 2, 0)
	return(temp)


def Find_Coin_Center(img):

	temp = cv.CloneImage(img)
	gray = cv.CreateImage(cv.GetSize(temp), 8, 1)	
	cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
	best_circle = (0,0,0)
	#print best_circle 
	#cv.Smooth(edges, edges, cv.CV_GAUSSIAN, 9, 9)
	for i in range (195, 220):
		#print i
		storage = cv.CreateMat(50, 1, cv.CV_32FC3)
		cv.SetZero(storage)
		kkk = cv.HoughCircles(gray, storage, cv.CV_HOUGH_GRADIENT, 1, float(40), float(175), float(55), long(i),long(350))

		num_of_circles = storage.rows

		for ii in range(num_of_circles):
			circle_data = storage[ii,0]
			center = cv.Round(circle_data[0]), cv.Round(circle_data[1])
			radius = cv.Round(circle_data[2])
			#print circle_data[0], circle_data[1], circle_data[2]
			if radius > 190:
				if radius > best_circle[2]:  
					#print "best was = ", best_circle
					best_circle = (circle_data[0], circle_data[1], circle_data[2])
					#print "best now = ", i				

 	return (best_circle)
	

img = cv.LoadImage(sys.argv[1])
coinimg_to_process = Find_Coin_Center(img)
#print coinimg_to_process
bounded_coin_img = Draw_Boundries(img, coinimg_to_process)

cv.ShowImage("Coin To Process",bounded_coin_img)
pil_img1 = Image.fromstring("L", cv.GetSize(img), img.tostring())
#pil_img1 = pil_img1.rotate(27, expand=True)
pil_img1 = pil_img1.rotate(27)
pil_img1.show()
cv_im = cv.CreateImageHeader(pil_img1.size, cv.IPL_DEPTH_8U, 1)
cv.SetData(cv_im, pil_img1.tostring())
cv.ShowImage("cv_im",cv_im)
cv.WaitKey()
Subtract_Images(img, cv_im)

cv.WaitKey()

