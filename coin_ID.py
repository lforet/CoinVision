#!/usr/bin/python

import sys
import easygui as eg
import cv, cv2
from threading import *
import time 
from img_processing_tools import *
from Tkinter import *

VIDEO_CAM_1 = 1
VIDEO_CAM_2 = 0
CROP_SIZE = 45
MOTOR_POWER = 25



class mywidgets:
	def __init__(self,root):
		frame=Frame(root)
		frame.pack()
		self.txtfr(frame)
		return

	def txtfr(self,frame):
		#define a new frame and put a text area in it
		self.textfr=Frame(frame)
		self.text=Text(self.textfr,height=10,width=50,background='white')
		# put a scroll bar in the frame
		scroll=Scrollbar(self.textfr)
		self.text.configure(yscrollcommand=scroll.set)
		
		#pack everything
		self.text.pack(side=LEFT)
		scroll.pack(side=RIGHT,fill=Y)
		self.textfr.pack(side=TOP)
		return

###########################################################
class log_display(Thread):
	def __init__(self):
		self.root = Tk()
		self.s = mywidgets(self.root)
		self.root.title('textarea')
		self.text = Text(self.root)
		
		Thread.__init__(self)

		def run(self):
			self.root.mainloop()


###########################################################
def get_new_coin(servo, dc_motor):
	coin_detection_threshold = 20
	servo.arm_down()
	base_frame = snap_shot(VIDEO_CAM_1)
	new_coin = False
	print 'CoinID Motor Driver Comm OPEN:', dc_motor.isOpen()
	print 'Connected to: ', dc_motor.portstr
	pilimg1 = CVtoPIL(CVtoGray(base_frame))
	print "grabbed image: ", pilimg1
	while not new_coin:
		if new_coin == False: move_motor(dc_motor, "F", MOTOR_POWER)
		if new_coin == False: time.sleep(.6)
		motor_stop(dc_motor)
		if new_coin == False: time.sleep(.9)
		frame = snap_shot(VIDEO_CAM_1)
		pilimg2 = CVtoPIL(CVtoGray(frame))
		rms_dif = rmsdiff(pilimg1, pilimg2)
		print "RMS Dif:", rms_dif 
		if rms_dif > coin_detection_threshold:
			print "New coin detected ...", rms_dif
			#sys.stdout.write('\a') #beep
			new_coin = True
###########################################################
		
def move_motor(dc_motor, direction, speed):
	if direction == "F":
		cmd_str = direction + str(speed) + '%\r'
		print cmd_str
		dc_motor.write ('GO\r')
		time.sleep(.01)
		dc_motor.write (cmd_str)
		time.sleep(.01)
		dc_motor.write ('GO\r')
		time.sleep(.01)

###########################################################
def motor_stop(dc_motor):
	dc_motor.write ('X\r\n')
###########################################################

def snap_shot(usb_device):
	print "Grabbing image from camera"
	#capture from camera at location 0
	#now = time.time()
	webcam1 = None
	frame = None
	#try:	
	while webcam1 == None:
		webcam1 = cv2.VideoCapture(usb_device)
		#webcam1 = cv.CreateCameraCapture(usb_device)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
		#time.sleep(.1)
	for i in range(3):
		ret, frame = webcam1.read()
		frame = array2cv(frame)
		#cv.GrabFrame(webcam1)
		#frame = cv.QueryFrame(webcam1)
	#except:
	#	print "******* Could not open WEBCAM *******"
	#	print "Unexpected error:", sys.exc_info()[0]
		#raise		
		#sys.exit(-1)
	#print frame
	#print webcam1
	#while webcam1 != None:
	cv2.VideoCapture(usb_device).release()
	return frame


###########################################################
class coincam_display(Thread):
	def __init__(self, Top_camID, Bottom_camID):
		self.Top_camID = Top_camID 
		self.Bottom_camID =  Bottom_camID
		self.img = None 
		Thread.__init__(self)

	def run(self):
		#self.img = cv2.imread('temp.png')
		#if self.img != None: print self.img
		cv2.namedWindow('Processing', cv.CV_WINDOW_AUTOSIZE)
		cv2.namedWindow('Top Cam', cv.CV_WINDOW_AUTOSIZE)
		cv2.namedWindow('Bottom Cam', cv.CV_WINDOW_AUTOSIZE)
		cv.MoveWindow('Processing', 40, 24)
		cv.MoveWindow('Top Cam', 460, 24)
		cv.MoveWindow('Bottom Cam', 860, 24)
		while True:
			time.sleep(.05)
			try:
				cv.ShowImage('Processing', self.img)
			except:
				pass
			try:
				frame = snap_shot(self.Top_camID)
				frame2 = snap_shot(self.Bottom_camID)
				time.sleep(.05)
				frame = resize_img(frame, 0.60)
				time.sleep(.05)
				frame2 = resize_img(frame2, 0.60)
				cv.ShowImage('Top Cam', frame)
				cv.ShowImage('Bottom Cam', frame2)
				cv.WaitKey(5)
			except:
				print "display failure"
				pass
###########################################################


if __name__=="__main__":

	coincam_disp = coincam_display(Top_camID=VIDEO_CAM_1, Bottom_camID=VIDEO_CAM_2)
	coincam_disp.daemon=True
	coincam_disp.start()
	time.sleep(3)

	my_log_display = log_display()
	my_log_display.daemon=True
	my_log_display.start()

	reply = ""
	while True:
		
		ready_to_display = False
		eg.rootWindowPosition = '+400+400'
		print eg.rootWindowPosition
		print 'reply=', reply		

		
		if reply == "Quit":
			print "Quitting...."
			sys.exit(-1)

		if reply == "Top Cam":
			img = snap_shot(VIDEO_CAM_1)
			img = resize_img(img, 0.60)
			coincam_disp.img = img

		if reply == "Bottom Cam":
			img = snap_shot(VIDEO_CAM_2)
			try:
				img = resize_img(img, 0.60)
				coincam_disp.img = img			
				#current_text = my_log_display.s.text.get(1.0, END)
				my_log_display.s.text.insert(INSERT,"Processing...\n")
			except:
				pass

		try:
			reply =	eg.buttonbox(msg='Coin Trainer', title='Coin Trainer', choices=('Top Cam', 'Bottom Cam', 'Quit'), image='', root=None)
		except:
			pass

