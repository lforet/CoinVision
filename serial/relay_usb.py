import serial
import time

serial_port = serial.Serial('/dev/ttyACM0')  # open first serial port


def Relay_ON(relay):
	if relay == "01":
		#set relay to listen
		serial_port.write(chr(0x5C))
		#set relay #2 OFF
		serial_port.write(chr(0x65))
		#set relay #1 ON
		serial_port.write(chr(0x70))
	if relay == "10":
		#set relay to listen
		serial_port.write(chr(0x5C))
		#set relay #1 OFF
		serial_port.write(chr(0x6F))
		#set relay #2 ON
		serial_port.write(chr(0x66))	
	if relay == "11":
		#set relay to listen
		serial_port.write(chr(0x5C))
		#set both relays ON
		serial_port.write(chr(0x65))
		serial_port.write(chr(0x65))
	if relay == "00":
		#set relay to listen
		serial_port.write(chr(0x5C))
		#set both relays OFF
		serial_port.write(chr(0x70))
		serial_port.write(chr(0x6F))


def Relay_Status():
	serial_port.write(chr(0x5B))
	serial_data = serial_port.read(1)
	result = int(ord(serial_data[0]))
	return result

serial_port.write(chr(0x5A))
s = serial_port.read(2)        # read up to ten bytes (timeout)
serial_port.write(chr(0x5C))
serial_port.write(chr(0x6E))
print "\n \n    USB-RLY02 - 2 relay outputs at 16A is found on port ", serial_port.portstr
print "Module id:", int(ord(s[0]))
print "Module software version: ",  int(ord(s[1]))
print "Relay States: ", Relay_Status()
print "wait 3 seconds.... \n"
time.sleep(3)

print "Turning both on..."
time.sleep(1)
Relay_ON("11")
time.sleep(1)
print "Relay States: ", Relay_Status()
print "wait 3 seconds....\n"
time.sleep(3)


print "Turning both off..."
time.sleep(1)
Relay_ON("00")
time.sleep(1)
print "Relay States: ", Relay_Status()
print "wait 3 seconds....\n"
time.sleep(3)

print "Turning relay #1 ON..."
time.sleep(1)
Relay_ON("01")
time.sleep(1)
print "Relay States: ", Relay_Status()
print "wait 3 seconds....\n"
time.sleep(3)

print "Turning relay #2 ON..."
time.sleep(1)
Relay_ON("10")
time.sleep(1)
print "Relay States: ", Relay_Status()
print "wait 3 seconds....\n"
time.sleep(3)

print "Turning both off..."
Relay_ON("00")
time.sleep(1)
print "Relay States: ", Relay_Status()

print "Going Crazzzzyyyy...\n"
time.sleep(3)
for a in range (0,20):
	time.sleep(0.05)
	Relay_ON("10")
	time.sleep(0.05)
	Relay_ON("01")

print "Turning both off...shutdown...test completed"
Relay_ON("00")

serial_port.close()             # close port

