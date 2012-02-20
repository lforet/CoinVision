import serial
import time
import sys


class PololuMicroMaestro(object):

	def __init__(self, port= "/dev/ttyACM1"):
		self.ser = serial.Serial(port = port)

	def LPF(self):
		print 'fffff'

	def setAngle(self, channel, angle):
		"""Set the target angle of the servo.  This is converted into "quarter microseconds", i.e., the pulse width necessary to get to that angle (and thus it's between 1.0ms and 2.0ms in increments of 0.25us).  Whew!"""
		minAngle = 0.0
		maxAngle = 180.0
		# these numbers, in quarter microseconds, taken from the code here:
		# http://forum.pololu.com/viewtopic.php?t=2380#p10697
		minTarget = 256.0
		maxTarget = 13120.0
		scaledValue = int((angle / ((maxAngle - minAngle) / (maxTarget - minTarget))) + minTarget)
		commandByte = chr(0x84)
		channelByte = chr(channel)
		lowTargetByte = chr(scaledValue & 0x7F)
		highTargetByte = chr((scaledValue >> 7) & 0x7F)
		command = commandByte + channelByte + lowTargetByte + highTargetByte
		self.ser.write(command)
		self.ser.flush()

##############################################################################################

	def setTarget(self,channel,value):
		#if not self.isInitialized: log("Not initialized"); return
		highbits,lowbits = divmod(value,32)
		#highTargetByte, lowTargetByte = divmod(value,32)
		#lowTargetByte = chr(value & 0x7F)
		#highTargetByte = chr((value >> 7) & 0x7F)
		#channelByte = chr(channel)
		#commandByte = chr(0x84)
		#command = commandByte + channelByte + lowTargetByte + highTargetByte
		self.write(0x84,channel,lowbits << 2,highbits)
		#self.ser.write(command)
		self.ser.flush()

#######################################################################
	def setSpeed(self, channel, speed):
		"""Set the speed of the given channel.  Speed is given in units of 0.25us / 10ms.  This means there is a range from 1 to 40000.  Getting a handle of what this _actually_ means in practice, in terms of the visual speed of the motors, will take a bit of work."""
		commandByte = chr(0x87)
		channelByte = chr(channel)
		lowTargetByte = chr(speed & 0x7F)
		highTargetByte = chr((speed >> 7) & 0x7F)
		command = commandByte + channelByte + lowTargetByte + highTargetByte
		self.ser.write(command)
		self.ser.flush()

#######################################################################
	def setAcceleration(self, channel, accel):
		"""Set the acceleration of this channel.  Value should be between 1 and 255.  A setting of 0 removes the acceleration limit."""
		commandByte = chr(0x89)
		channelByte = chr(channel)
		lowTargetByte = chr(accel)
		highTargetByte = chr(0x00)
		command = commandByte + channelByte + lowTargetByte + highTargetByte
		self.ser.write(command)
		self.ser.flush()
#######################################################################
	def getPosition(self, channel):
		"""Get the position of this servo.  Returned in units of us."""
		commandByte = chr(0x90)
		channelByte = chr(channel)
		command = commandByte + channelByte
		self.ser.write(command)
		lowByte = ord(self.ser.read(1))
		highByte = ord(self.ser.read(1))
		highByte = highByte << 8
		position = highByte + lowByte
		return (position / 4.0)
#######################################################################
	def goHome(self):
		"""Set all servos to home position."""
		self.ser.write(chr(0xA2))
#######################################################################
	def close(self):
		self.ser.close()
    ###########################################################################################################################
	## common write function for handling all write related tasks
	def write(self,*data):
		#if not self.isInitialized: log("Not initialized"); return
		#if not self.ser.writable():
		#    log("Device not writable")
		#    return
		for d in data:
		    self.ser.write(chr(d))
		self.ser.flush()

if __name__=="__main__":

	#serial_port = serial.Serial(port='/dev/ttyACM1', baudrate=9600)

	d = PololuMicroMaestro()

	print d
	
	print  d.getPosition(0)
	#d.setAngle(0,80)
	#d.setTarget(0,1500)
	print d.setSpeed(0,30)
	print d.LPF()


	d.setTarget(0,1500)
	#print d.goHome()
	#serial_port.write(chr(0x80))
	#time.sleep(.1)
	#print serial_port.write(chr(0xA2))
	#time.sleep(.1)
	#serial_port.flush()

	#print serial_port.write(chr(0x90)+chr(0x00))
	#serial_port.flush()


	#serial_port = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=1)
	#serial_port.open()
	#print serial_port.isOpen()


	time.sleep(1)

	#compact protocol
	#serial_port.write(chr(0x84)+chr(0x00)+chr(0x70)+chr(0x2E))
	#serial_port.write(chr(0xAA)+chr(0x0C)+chr(0x04)+chr(0x00)+chr(0x08)+chr(0x27))

	#set_target(0,1750)
	#time.sleep(1)
	#serial_port.close()

	del d
