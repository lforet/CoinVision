#!/usr/bin/python

# iterate through a directroy
#	load image file#
#	if neccessary convert to greyscale
#	rotate image -45  to +45 degrees
#	append class data file with image

import sys
import opencv
import pil



def gray_images(img):

	temp_img = cv.CreateImage(cv.GetSize(img), 8, 1)
	if img.channels == 1:
		temp_img = img
	if img.channels > 1:
		cv.CvtColor(img, temp_img, cv.CV_BGR2GRAY)
	return(temp_img)

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

if __name__=="__main__":
	
	#if len(sys.argv) < 3:
	#	print "******* Requires 2 image files for comparison. *******"
	#	sys.exit(-1)
	if len(sys.argv) == 1:
			path='training_images'
		
	if len(sys.argv) == 2:
			path= sys.argv[1]
	print "\n Attempting to process files in directory: ", os.getcwd()+"/"+ path


	count = 0
	for subdir, dirs, files in os.walk(path):
		count = len(files)
	if count == 0:
		print "No files in directory to process..."
		sys.exit(-1)
	if count > 0:
		#delete classid and classdata files to completely rebuild them 
		f_handle = open("greenbandclassid.txt", 'w')
		f_handle.close()
		f_handle = open("greenbanddata.txt", 'w')
		f_handle.close()
		print "Files to Process: ", count
		for subdir, dirs, files in os.walk(path):
			for file in files:
				filename1= os.path.join(path, file)
				try:
					img1 = cv.LoadImage(filename1)
				except:
					print "******* Could not open image files *******"
					sys.exit(-1)
				print "\n Processing current image: " , filename1 
			
			if im.size[0] <> 40 or im.size[1] <> 40:
				print "Image is not right size. Resizing image...."
				im = im.resize((320, 240))
				print "Resized to 320, 340"
			if im.mode == "RGB":
				print "Image has multiple color bands...Splitting Bands...."
				Red_Band, Green_Band,Blue_Band = im.split()
				
				im.show()
				print Green_Band
				Green_Band.show()
				#print "Saving color bands...."
				#filename = filename1.rsplit('.')[0] + "_RedBand.bmp"
				#print filename1.rsplit('.')[0][-1]
				imageclassid = filename1.rsplit('.')[0][-1]
				classid = array(int(imageclassid[0]))
				if imageclassid.isdigit():
					print "Image class: ", imageclassid
					f_handle = open("greenbandclassid.txt", 'a')
					f_handle.write(str(classid))
					f_handle.write(' ')
					f_handle.close()
					
					#calculate histogram
					print "Calculating Histogram for the green pixels of image..."
					Histogram = CalcHistogram(Green_Band)
					#save Green Histogram to file in certain format
					print "saving histogram to dictionary..."
					f_handle = open("greenbanddata.txt", 'a')
					for i in range(len(Histogram)):
						f_handle.write(str(Histogram[i]))
						f_handle.write(" ")
					f_handle.write('\n')
					f_handle.close()
				#print "Saving.....", filename 
				#Red_Band.save(filename, "BMP")
				#filename = filename1.rsplit('.')[0] + "_GreenBand.bmp"
					print "calling i3"
					I3image = rgb2I3(im)
					#calculate histogram
					print "Calculating Histogram for I3 pixels of image..."
					Red_Band, Green_Band, Blue_Band = I3image.split()
					Histogram = CalcHistogram(Green_Band)
					#save I3 Histogram to file in certain format
					f_handle = open("I3bandclassid.txt", 'a')
					f_handle.write(str(classid))
					f_handle.write(' ')
					f_handle.close()
					print "saving I3 histogram to dictionary..."
					f_handle = open("I3banddata.txt", 'a')
					for i in range(len(Histogram)):
						f_handle.write(str(Histogram[i]))
						f_handle.write(" ")
					f_handle.write('\n')
					f_handle.close()


