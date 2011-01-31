import sys, os
import cv
import Image
import time
import scipy.spatial

#############################################################################
# some "global" variables

image = None
pt1 = (-1,-1)
pt2 = (-1,-1)
add_remove_pt = False
flags = 0
night_mode = False
need_to_init = False

#############################################################################
# the mouse callback

# the callback on the trackbar
def on_mouse (event, x, y, flags, param):

	# we will use the global pt and add_remove_pt
	global pt1
	global pt2

	img1_copy = cv.CloneImage(param)

	if event == cv.CV_EVENT_LBUTTONDOWN:
		# user has click, so memorize it
		if (pt1[0] > 0) & (pt2[0] > 0):
			pt1 = (-1, -1)
			pt2 = (-1, -1)
			cv.ShowImage("After Scale img1", img1_copy)
			print "both >"
		elif pt1[0] == -1 :
			pt1 = (x, y)
			pt2 = (-1, -1)
			print "pt 1"
		elif (pt1[0] > 0) & (x > pt1[0]) & (y > pt1[1]): 
			pt2 = (x, y)	
			#Draw date bounding rectangle  q		
			cv.Rectangle(img1_copy , pt1, pt2, cv.CV_RGB(255, 255, 0), 2, 0)
			cv.ShowImage("After Scale img1", img1_copy)
		print "pt1, pt2 = ", pt1,pt2
		
	#if x+y > 1:
	#pt1 = (x, y)
	#	cv.Circle(img1_copy,pt, 2, cv.CV_RGB(0, 0, 255), 2, cv.CV_AA, 0 ) 
	#	cv.ShowImage("Coin Image 1",img1_copy)
	#	print pt
		#add_remove_pt = True

#############################################################################

def rotate_image(img, degrees):
	"""
    rotate(scr1, degrees) -> image
    Parameters:	

         *  image - source image
         *  angle (integer) - The rotation angle in degrees. Positive values mean counter-clockwise 	rotation 
	"""
	temp_img = cv.CreateImage(cv.GetSize(img), 8, img.channels)
	mapMatrix = cv.CreateMat( 2, 3, cv.CV_32FC1 )
	img_size = cv.GetSize(img)
	img_center = (int(img_size[0]/2), int(img_size[1]/2))
	cv.GetRotationMatrix2D(img_center, degrees, 1.0, mapMatrix)
	cv.WarpAffine(img , temp_img, mapMatrix, flags=cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))
	return(temp_img)

def get_image( camera1 ):
	img = cv.QueryFrame( camera1 )
	return img

def draw_date_boundry(img, point1, point2):

	cv.Rectangle(img, point1, point2, cv.CV_RGB(255, 255, 0), 2, 0) 
	cv.ShowImage("temp_img", img)

def scale_and_crop(img1, img2):
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
	#topleft_corner1 = (coin1_center[0]-coin1_radius-size_buffer, coin1_center[1]-coin1_radius-size_buffer)
	#bottomright_corner1 = (coin1_center[0]+coin1_radius+size_buffer, coin1_center[1]+coin1_radius+size_buffer)
	#topleft_corner2 = (coin2_center[0]-coin2_radius-size_buffer, coin2_center[1]-coin2_radius-size_buffer)
	#bottomright_corner2 = (coin2_center[0]+coin2_radius+size_buffer, coin2_center[1]+coin2_radius+size_buffer)
	#crop inside bounding rectangle for orientation 
	topleft_corner1 = (coin1_center[0]-int((coin1_inside_radius*(cv.Sqrt(2)/2))), coin1_center[1]-int((coin1_inside_radius*(cv.Sqrt(2)/2))))
	bottomright_corner1 = (coin1_center[0]+int((coin1_inside_radius*(cv.Sqrt(2)/2))), coin1_center[1]+int((coin1_inside_radius*(cv.Sqrt(2)/2))))
	topleft_corner2 = (coin2_center[0]-int((coin2_inside_radius*(cv.Sqrt(2)/2))), coin2_center[1]-int((coin2_inside_radius*(cv.Sqrt(2)/2))))
	bottomright_corner2 = (coin2_center[0]+int((coin2_inside_radius*(cv.Sqrt(2)/2))), coin2_center[1]+int((coin2_inside_radius*(cv.Sqrt(2)/2))))

	cropped_img1 = cv.GetSubRect(img1, (topleft_corner1[0], topleft_corner1[1], bottomright_corner1[0]-topleft_corner1[0], bottomright_corner1[1]-topleft_corner1[1]))
	cropped_img2 = cv.GetSubRect(img2, (topleft_corner2[0], topleft_corner2[1], bottomright_corner2[0]-topleft_corner2[0], bottomright_corner2[1]-topleft_corner2[1]))

	print "Before resize SIZES = ", cv.GetSize(cropped_img1), cv.GetSize(cropped_img2)
	temp_img = cv.CreateImage(cv.GetSize(cropped_img1), 8, img2.channels)
	temp_img2 = cv.CreateImage(cv.GetSize(cropped_img1), 8, img1.channels)
	cv.Resize(cropped_img2, temp_img)
	cv.Resize(cropped_img1, temp_img2)
	print "Before resize SIZES = ", cv.GetSize(cropped_img1), cv.GetSize(temp_img)
	#cv.WaitKey()
	return(temp_img2, temp_img)

def gray_images(img):

	temp_img = cv.CreateImage(cv.GetSize(img), 8, 1)
	if img.channels == 1:
		temp_img = img
	if img1.channels > 1:
		cv.CvtColor(img, temp_img, cv.CV_BGR2GRAY)
	return(temp_img)


def get_orientation(img1, img2):

	subtracted_image = cv.CreateImage(cv.GetSize(img1), 8, 1)
	temp_img = cv.CreateImage(cv.GetSize(img1), 8, 1)

	img1_copy = cv.CloneImage(img1)	
	img2_copy = cv.CloneImage(img2)
	canny_parm1 = 87
	canny_parm2 = 175
	to_smooth = 1
	very_best_sum = 0
	very_best_orientation = 0
	best_settings = [0,0,0,0,0]

	#for canny_parm1 in range(125,40, - 1):	
	#	for canny_parm2 in range(250, 40, - 1):
	#		print "iteration = ", canny_parm1 , canny_parm2, "  Best Settings = ",best_settings 
	#		for to_smooth in range(1,3):
				#print "settings = ", to_smooth
	if to_smooth == 1:
		cv.Smooth(img1_copy, img1_copy, cv.CV_GAUSSIAN, 3, 3)
		cv.Smooth(img2_copy, img2_copy, cv.CV_GAUSSIAN, 3, 3)
	#cv.WaitKey()

	cv.Canny(img1_copy,img1_copy ,canny_parm1,canny_parm2, 3)
	cv.Canny(img2_copy,img2_copy, canny_parm1,canny_parm2, 3)
	cv.ShowImage("img1", img1_copy)
	cv.ShowImage("img2", img2_copy)

	temp_img = rotate_image(img2, very_best_orientation)
	cv.ShowImage("corrected img2", temp_img)
	#cv.WaitKey()
	best_sum = 0
	best_orientation = 0
	best_euclidean = 0
	best_orientation_euclidean = 0
	for i in range(1, 360):
		temp_img = rotate_image(img2_copy, i)
		cv.And(img1_copy, temp_img , subtracted_image)
		# cv.ShowImage("subtracted_image", subtracted_image)
		#cv.ShowImage("Image of Interest", temp_img )
		sum_of_and = cv.Sum(subtracted_image)
		if best_sum == 0: best_sum = sum_of_and[0]
		if sum_of_and[0] > best_sum: 
			best_sum = sum_of_and[0]
			best_orientation = i
		#print i, "Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "  best_orientation =", best_orientation
		e_dist = scipy.spatial.distance.euclidean(cv.GetMat(img1_copy), cv.GetMat(temp_img))
		if best_euclidean  == 0: best_euclidean = e_dist
		if e_dist < best_euclidean: 
			print "best_euclidean =", e_dist, i
			best_euclidean =  e_dist
			best_orientation_euclidean = i
			cv.ShowImage("Image of Interest", temp_img )
			#cv.WaitKey()
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break
		#time.sleep(.01)
		if (best_sum > very_best_sum): #& (best_orientation > 120) & (best_orientation < 125):
			very_best_sum = best_sum
			very_best_orientation = best_orientation
			best_settings = [canny_parm1 ,canny_parm2 ,to_smooth, very_best_sum, very_best_orientation]
			print "New Best Settings = ",best_settings 

	img1_copy = cv.CloneImage(img1)	
	img2_copy = cv.CloneImage(img2)

	print "Final Best Settings = ", best_settings
	print "best_orientation_euclidean = ", best_orientation_euclidean
	return (very_best_orientation)


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
	for i in range (180, 235):
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
			if radius > 180:
				if radius > best_circle[2]:  
					#print "best was = ", best_circle
					best_circle = (circle_data[0], circle_data[1], circle_data[2])
					#print "best now = ", i				

 	return (best_circle)




if __name__=="__main__":
	
	if len(sys.argv) < 3:
		print "******* Requires 2 image files for comparison. *******"
		sys.exit(-1)

	try:
		img1 = cv.LoadImage(sys.argv[1])
		img2 = cv.LoadImage(sys.argv[2])
	except:
		print "******* Could not open image files *******"
		sys.exit(-1)

	bounded_coin_img1 = draw_boundries(img1)
	bounded_coin_img2 = draw_boundries(img2)
	

	cv.ShowImage("Coin Image 1",bounded_coin_img1)
	cv.ShowImage("Coin Image 2",bounded_coin_img2)
	cv.WaitKey()
	img1_copy = cv.CloneImage(img1)
	img2_copy = cv.CloneImage(img2)

	img1_copy, img2_copy = scale_and_crop(img1, img2)
	cv.ShowImage("After Scale img1", img1_copy)
	cv.ShowImage("After Scale img2", img2_copy)

	# register the mouse callback
	cv.SetMouseCallback ("After Scale img1", on_mouse, img1_copy)
	
	camera1 = cv.CreateCameraCapture( 0 )
	
	while True:
		c = cv.WaitKey(5)
		if c == 27 or c == ord('q') or c == 1048688 or c == 1048603:
		    break
		if c == ord('p'):
			break
		if c == ord('s'):
			 img1_copy = get_image(camera1)
			 cv.ShowImage("camera display", img1_copy)
		if c != -1:
			print c

	
	img1_gray = cv.CloneImage(img1_copy)
	img2_gray = cv.CloneImage(img2_copy)

	img1_gray = gray_images(img1_gray)
	img2_gray = gray_images(img2_gray)

	#cv.ShowImage("after grey img1", img1_gray)
	#cv.ShowImage("after grey img2", img2_gray)
	#cv.WaitKey()
	coin_orientation = get_orientation(img1_gray, img2_gray)
	print "The coin is offest ", coin_orientation, " degrees"

	img2 = rotate_image(img2, coin_orientation)

	draw_date_boundry(img2, pt1, pt2)

	cv.ShowImage("after all img1", img1)
	cv.ShowImage("after all img2", img2)
	cv.WaitKey()


