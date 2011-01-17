import serial
import time

serial_port = serial.Serial(port='/dev/ttyUSB1', baudrate=115200, timeout=None)
serial_port.open()
print serial_port.isOpen()

print 'Lynxmotion SSC32 Python interface'
print 'Connected to: ', serial_port.portstr


time.sleep(2)
print "sending command"

def arm_move(move):
	if move == "straight_up":
		serial_port.write ('#1 P1300 #2 P1300 #3 P600 #4 P1630 T1500\r\n')
	if move == "inital_position_1":
		serial_port.write ('#0 P1750 #1 P1100 #2 P1100 #3 P1500 #4 P550 #5 P 1850 T1500\r\n')
	if move == "safety_position_1":
		serial_port.write ('#0 P1750 #1 P575 #2 P575 #3 P2050 #4 P1630 T1500\r\n')

def arm_pause(secs):
	time.sleep(secs)


while 1:
	 firmware = ''
	 serial_port.write('ver\r\n')
	 time.sleep(0.05)
	 while serial_port.inWaiting() > 0:
	    firmware += serial_port.read()

	 if firmware == '': firmware = 'ver'

	 input = raw_input(firmware.strip() + ' >>>')

         if input == 'quit':
            print "Press 'ESC' to quit or 'F1' to enter shellmode"
            break

         if (input != 'ver') & (input != 'quit') & (input != 'mm'):
		#ser.write(input + '\r\n')
		print "Sending command: ", input
		serial_port.write (input+'\r\n')

         if input == 'ver':
		#ser.write(input + '\r\n')
		serial_port.write ('ver\r\n')
		time.sleep(0.05)
		data = ""
		while serial_port.inWaiting() > 0:
			data += serial_port.read()
		print data

         if input == 'mm':
		#ser.write(input + '\r\n')
		serial_port.write ('#0 P1750 #1 P1100 #2 P1100 #3 P1500 #4 P550 #5 P 1850 T1500\r\n')
		time.sleep(2)

		serial_port.write ('#0 P2200 #1 P1400 #2 P1400 #3 P1000 #4 P550 T1200\r\n')

		time.sleep(1.5)

		serial_port.write ('#5 P700\r\n')
		time.sleep(.8)
		serial_port.write ('#1 P1650 #2 P1650 #3 P1600 #4 P1150 T400\r\n')
		time.sleep(.3)
		serial_port.write ('#3 P1500 #4 P1200 #5 P700 \r\n')
		time.sleep(.5)
		serial_port.write ('#3 P1400 #5 P1800 \r\n')
		time.sleep(.5)
		serial_port.write ('#3 P1600 #5 P700 \r\n')
		time.sleep(.4)
		serial_port.write ('#1 P1675 #2 P1675 #3 P1650 #4 P1100 \r\n')
		time.sleep(.5)
		serial_port.write ('#4 P900\r\n')
		time.sleep(.2)
		serial_port.write ('#5 P1900\r\n')
		time.sleep(1)
		serial_port.write ('#1 P1675 #2 P1675 #3 P1650 #4 P1300 T1400 \r\n')
		time.sleep(.4)
		serial_port.write ('#1 P1400 #2 P1400 #3 P1250 #4 P550 T200\r\n')
		time.sleep(.5)
		serial_port.write ('#1 P1200 #2 P1200 T500\r\n')
		time.sleep(.6)
		serial_port.write ('#0 P1850 T1000\r\n')
		time.sleep(1.5)
		serial_port.write ('#0 P1750 #1 P1100 #2 P1100 #3 P1500 #4 P550 #5 P 1900 T1500\r\n')
		time.sleep(.8)
		serial_port.write ('#0 P1850 #1 P1050 #2 P1050 #3 P1000 #4 P1000 T1500\r\n')
		time.sleep(1)
		serial_port.write ('#1 P1350 #2 P1350 #3 P1000 T1000\r\n')
		time.sleep(1)
		serial_port.write ('#5 P1000\r\n')
		time.sleep(1)
		serial_port.write ('#3 P1000 #4 P2000 #1 P1200 #2 P1200 T1000\r\n')
		time.sleep(1)		
		serial_port.write ('#4 P550 T500\r\n')
		time.sleep(.5)
		serial_port.write ('#0 P1750 #1 P1100 #2 P1100 #3 P1500 #4 P550 #5 P 1850 T1500\r\n')
#s = serial_port.readline()        # read up to ten bytes (timeout)
#print s


serial_port.close()             # close port



"""
#0 P1750 = center arm facing forward
#0 P2500 all right
#0 P800 all left
#1 P1300 #2 P1300 Shoulder Straight up
#1 P650 #2 P650 Shoulder all back
#1 P2200 #2 P2200
#3 P600 Straight Elbow 
#3 P 2000 Elbow all down
#3 P1330 = 90 deg
#4 P550 = wrist all down
#4 P2500 All up
#4 P1630 = straight
#5 P1850 Total Close
#5 P700 Total Open

#0 P2250 #1 P1600 #2 P1600 #3 P1450 #4 P1200  T2500


#1 P650 #2 P650 
"""
#1 P1500 #2 P1500 T2000

