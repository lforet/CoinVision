import sys, os
import cv
import Image
import time


def draw_date_boundry(img, offset):

	temp_img = cv.CreateImage(cv.GetSize(img), 8, img.channels)
	
	mapMatrix = cv.CreateMat( 2, 3, cv.CV_32FC1 )
	img_center = (cv.GetSize(img)[0]/2, cv.GetSize(img)[1]/2)
	cv.GetRotationMatrix2D(img_center, offset, 1.0, mapMatrix)
	cv.WarpAffine(img , temp_img, mapMatrix, flags=cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))

	size_buffer = 15
	radius_buffer = 50

	coin1 = get_coin_center(img)
	print "draw date coin1, coin2 = ", coin1  
	coin1_center = int(coin1[0]), int(coin1[1])
	coin1_radius = int(coin1[2])
	coin1_inside_radius = coin1_radius - radius_buffer
	
	#Draw date bounding rectangle  
	topleft_corner = (coin1_center[0]+coin1_inside_radius- size_buffer, coin1_center[1])
	bottomright_corner = (coin1_center[0]+coin1_radius+size_buffer, coin1_center[1]+coin1_inside_radius- size_buffer )
	cv.Rectangle(temp_img, topleft_corner, bottomright_corner, cv.CV_RGB(255, 255, 0), 2, 0)
	cv.Circle(temp_img ,(coin1_center), 2, cv.CV_RGB(255, 255, 255), 2, cv.CV_AA, 0 ) 
	cv.ShowImage("temp_img", temp_img)

	

def get_orientation(img1, img2):
	size_buffer = 15
	radius_buffer = 50
	coin1 = get_coin_center(img1)
	coin2 = get_coin_center(img2)
	print "coin1, coin2 = ", coin1, coin2 
	print coin1[2]-coin2[2]
	 
	coin1_center = int(coin1[0]), int(coin1[1])
	coin1_radius = int(coin1[2])
	coin1_inside_radius = coin1_radius - radius_buffer
	coin2_center = int(coin2[0]), int(coin2[1])
	coin2_radius = int(coin2[2])
	coin2_inside_radius = coin2_radius - radius_buffer

	#crop OUTSIDE bounding rectangle for orientation 
	topleft_corner1 = (coin1_center[0]-coin1_radius-size_buffer, coin1_center[1]-coin1_radius-size_buffer)
	bottomright_corner1 = (coin1_center[0]+coin1_radius+size_buffer, coin1_center[1]+coin1_radius+size_buffer)
	topleft_corner2 = (coin2_center[0]-coin2_radius-size_buffer, coin2_center[1]-coin2_radius-size_buffer)
	bottomright_corner2 = (coin2_center[0]+coin2_radius+size_buffer, coin2_center[1]+coin2_radius+size_buffer)
	#crop inside bounding rectangle for orientation 
	#topleft_corner1 = (coin1_center[0]-int((coin1_inside_radius*(cv.Sqrt(2)/2))), coin1_center[1]-int((coin1_inside_radius*(cv.Sqrt(2)/2))))
	#bottomright_corner1 = (coin1_center[0]+int((coin1_inside_radius*(cv.Sqrt(2)/2))), coin1_center[1]+int((coin1_inside_radius*(cv.Sqrt(2)/2))))
	#topleft_corner2 = (coin2_center[0]-int((coin2_inside_radius*(cv.Sqrt(2)/2))), coin2_center[1]-int((coin2_inside_radius*(cv.Sqrt(2)/2))))
	#bottomright_corner2 = (coin2_center[0]+int((coin2_inside_radius*(cv.Sqrt(2)/2))), coin2_center[1]+int((coin2_inside_radius*(cv.Sqrt(2)/2))))

	cropped_img1 = cv.GetSubRect(img1, (topleft_corner1[0], topleft_corner1[1], bottomright_corner1[0]-topleft_corner1[0], bottomright_corner1[1]-topleft_corner1[1]))

	cropped_img2 = cv.GetSubRect(img2, (topleft_corner2[0], topleft_corner2[1], bottomright_corner2[0]-topleft_corner2[0], bottomright_corner2[1]-topleft_corner2[1]))

	#cv.ShowImage("cropped_img1 ", cropped_img1 )
	#cv.ShowImage("cropped_img2 ", cropped_img2 )
	print "Before reize SIZES = ", cv.GetSize(cropped_img1), cv.GetSize(cropped_img2)
	

	gray1 = cv.CreateImage(cv.GetSize(cropped_img1), 8, 1)
	gray2 = cv.CreateImage(cv.GetSize(cropped_img1), 8, 1)
	temp_img2 = cv.CreateImage(cv.GetSize(cropped_img1), 8, 3)
	temp_img = cv.CreateImage(cv.GetSize(cropped_img1), 8, 1)
	subtracted_image = cv.CreateImage(cv.GetSize(cropped_img1), 8, 1)
	cv.Resize(cropped_img2, temp_img2)
	cv.CvtColor(cropped_img1, gray1, cv.CV_BGR2GRAY)
	cv.CvtColor(temp_img2, gray2, cv.CV_BGR2GRAY)
	cv.ShowImage("gray1", gray1)
	cv.ShowImage("gray2", gray2)
	print "After Resize SIZES = ", cv.GetSize(gray1), cv.GetSize(gray2)
	cv.WaitKey()

	cv.Smooth(gray1, gray1, cv.CV_GAUSSIAN, 9, 9)
	cv.Smooth(gray2, gray2, cv.CV_GAUSSIAN, 9, 9)

	cv.Canny(gray1,gray1 ,87,175, 3)
	cv.Canny(gray2,gray2, 87,175, 3)
	cv.ShowImage("gray1", gray1)
	cv.ShowImage("gray2", gray2)
	cv.WaitKey()
	best_sum = 0
	best_orientation = 0
	for i in range(1, 360):
		mapMatrix = cv.CreateMat( 2, 3, cv.CV_32FC1 )
		center = (cv.GetSize(gray2 )[0]/2, cv.GetSize(gray2 )[1]/2)
		cv.GetRotationMatrix2D(center, i, 1.0, mapMatrix)
		cv.WarpAffine(gray2 , temp_img, mapMatrix, flags=cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))
		cv.And(gray1,temp_img, subtracted_image)
		cv.ShowImage("subtracted_image", subtracted_image)
		cv.ShowImage("temp_img", temp_img)
		sum_of_and = cv.Sum(subtracted_image)
		if best_sum == 0: best_sum = sum_of_and[0]
		if sum_of_and[0] > best_sum: 
			best_sum = sum_of_and[0]
			best_orientation = i
		#print i, "Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "  best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key ==1048603: break
		time.sleep(.01)
	draw_date_boundry(gray2, best_orientation)
	cv.WaitKey()
	return (best_orientation)


def draw_boundries(img):

	size_buffer = 15
	radius_buffer = 50
	coin_center = get_coin_center(img)
	center = int(coin_center[0]), int(coin_center[1])
	radius = int(cv.Round(coin_center[2]))
	inside_radius = radius - radius_buffer
	print coin_center 
	temp = cv.CloneImage(img)
	cv.Circle(temp, (center), radius, cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 )
	cv.Circle(temp ,(center), 2, cv.CV_RGB(0, 0, 255), 2, cv.CV_AA, 0 ) 	
	cv.Circle(temp ,(center), (radius - radius_buffer), cv.CV_RGB(0, 0, 255), 2, cv.CV_AA, 0 )

	#Draw outside bounding rectangle 
	topleft_corner = (center[0]-radius-size_buffer, center[1]-radius-size_buffer)
	bottomright_corner = (center[0]+radius+size_buffer, center[1]+radius+size_buffer)
	cv.Rectangle(temp, topleft_corner, bottomright_corner, cv.CV_RGB(255, 255, 0), 2, 0)
	#Draw inside bounding rectangle 
	topleft_corner = (center[0]-int((inside_radius*(cv.Sqrt(2)/2))), center[1]-int((inside_radius*(cv.Sqrt(2)/2))))
	bottomright_corner = (center[0]+int((inside_radius*(cv.Sqrt(2)/2))), center[1]+int((inside_radius*(cv.Sqrt(2)/2))))
	cv.Rectangle(temp, topleft_corner, bottomright_corner, cv.CV_RGB(255, 255, 0), 2, 0)
	return(temp)


def get_coin_center(img):

	temp = cv.CloneImage(img)
	gray = cv.CreateImage(cv.GetSize(temp), 8, 1)	
	if img.channels != 1: cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
	best_circle = (0,0,0)
	#print best_circle 
	#cv.Smooth(edges, edges, cv.CV_GAUSSIAN, 9, 9)

	for i in range (185, 225):
		#print i
		storage = cv.CreateMat(50, 1, cv.CV_32FC3)
		cv.SetZero(storage)
		cv.HoughCircles(gray, storage, cv.CV_HOUGH_GRADIENT, 1, float(40), float(175), float(55), long(i),long(230))

		num_of_circles = storage.rows
		
		for ii in range(num_of_circles):
			circle_data = storage[ii,0]
			center = cv.Round(circle_data[0]), cv.Round(circle_data[1])
			radius = cv.Round(circle_data[2])
			#print circle_data[0], circle_data[1], circle_data[2]
			if radius > 185:
				if radius > best_circle[2]:  
					#print "best was = ", best_circle
					best_circle = (circle_data[0], circle_data[1], circle_data[2])
					#print "best now = ", i				

 	return (best_circle)




	
img = cv.LoadImage(sys.argv[1])
pil_img1 = Image.open(sys.argv[2])
#pil_img1 = pil_img1.rotate(45)
cv_im = cv.CreateImageHeader(pil_img1.size, cv.IPL_DEPTH_8U, 3)
cv.SetData(cv_im, pil_img1.tostring())

bounded_coin_img1 = draw_boundries(img)
bounded_coin_img2 = draw_boundries(cv_im)

cv.ShowImage("Coin Image 1",bounded_coin_img1)
cv.ShowImage("Coin Image 2",bounded_coin_img2)

coin_orientation= get_orientation(img, cv_im)
print "The coin is offest ", coin_orientation, " degrees"
draw_date_boundry(cv_im, coin_orientation)

cv.WaitKey()

