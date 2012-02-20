import serial
import time
import sys


serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)

def move_motor(direction, speed):
	if direction == "F":
		cmd_str = direction + str(speed) + '%\r'
		print cmd_str
		serial_port.write ('GO\r')
		time.sleep(.1)
		serial_port.write (cmd_str)
		time.sleep(.1)
		serial_port.write ('GO\r')
		time.sleep(.1)
		#print serial_port.read()
		#print "F10%"
		#time.sleep(.1)
		#serial_port.write ('GO\r')
		#time.sleep(.1)
		#print serial_port.read()
def motor_stop():
	serial_port.write ('X\r\n')

def arm_pause(secs):
	time.sleep(secs)


if __name__=="__main__":

	#serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=1)
	serial_port.open()
	print serial_port.isOpen()
	firmware = ''
	serial_port.write('V\r')
	time.sleep(0.05)
	while serial_port.inWaiting() > 0:
		firmware += serial_port.read()

	print "firmware=", firmware
	
	#sys.exit(-1)

	time.sleep(0.1)
	print 'CoinID Motor Driver Comm OPEN:', serial_port.isOpen()
	print 'Connected to: ', serial_port.portstr

	print "send command"
	move_motor("F", 15)
	time.sleep(10)
	print "stopping"
	motor_stop()
	time.sleep(1)

	temp = ''
	serial_port.write('D24\r')
	time.sleep(0.05)
	while serial_port.inWaiting() > 0:
		temp += serial_port.read()

	print "temp=", temp




	serial_port.close()
