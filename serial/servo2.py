#!/usr/bin/python
# Lynxmotion SSC32 Python interface
# by rakware @2008, mail stuff to cacatpunctro at yahoo dot com
# serial module from http://pyserial.wiki.sourceforge.net/pySerial
# _Getch class from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/134892
# please change the com port to suit your needs.
# WARNING! : code provided "AS IS", use at your own risk.

import time
import serial

class _Getch:
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()

ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=None)
ser.open()
ser.isOpen()

#to do: add SerialException with error message if port could not be open

print 'Lynxmotion SSC32 Python interface'
print 'Connected to: ', ser.portstr
print "Press 'ESC' to quit or 'F1' to enter shellmode"

while 1:
   char = getch()
   if char == chr(27): #27 is ESC

      ser.close()
      movementr.close()
      movementa.close()
      runtimew.close()
      runtimer.close()
      exit()
   elif char == chr(13):
      print
   elif char == chr(59): #59 is F1
      print "Type 'quit' to return"
      while 1:
         firmware = ''
         ser.write('ver\r\n')
         time.sleep(0.05)
         while ser.inWaiting() > 0:
            firmware += ser.read()
         if firmware == '': firmware = 'ver'
   
         input = raw_input(firmware.strip() + ' >>>')

         if input == 'quit':
            print "Press 'ESC' to quit or 'F1' to enter shellmode"
            break

         elif input == 'va':
            ser.write(input + '\r\n')
            out = ''
            time.sleep(0.05)
            while ser.inWaiting() > 0:
               out += ser.read()
         
            if out != '':
               #print round(ord(out) *5/256.0, 3)
               print ord(out)

         elif input == 'vb':
            ser.write(input + '\r\n')
            out = ''
            time.sleep(0.05)
            while ser.inWaiting() > 0:
               out += ser.read()
         
            if out != '':
               #print round(ord(out) *5/256.0, 3)
               print ord(out)

         elif input == 'vc':
            ser.write(input + '\r\n')
            out = ''
            time.sleep(0.05)
            while ser.inWaiting() > 0:
               out += ser.read()
         
            if out != '':
               #print round(ord(out) *5/256.0, 3)
               print ord(out)

         elif input == 'vd':
            ser.write(input + '\r\n')
            out = ''
            time.sleep(0.05)
            while ser.inWaiting() > 0:
               out += ser.read()
         
            if out != '':
               #print round(ord(out) *5/256.0, 3)
               print ord(out)

         elif input == 'va vb':
            ser.write(input + '\r\n')
            out = ''
            time.sleep(0.05)
            while ser.inWaiting() > 0:
               out += ser.read()
         
            if out != '':
               print ord(out[:1]), ord(out[1:2])

         elif input == 'va vb vc':
            ser.write(input + '\r\n')
            out = ''
            time.sleep(0.05)
            while ser.inWaiting() > 0:
               out += ser.read()
         
            if out != '':
               print ord(out[:1]), ord(out[1:2]), ord(out[2:3])

         elif input == 'va vb vc vd':
            ser.write(input + '\r\n')
            out = ''
            time.sleep(0.05)
            while ser.inWaiting() > 0:
               out += ser.read()
         
            if out != '':
               print ord(out[:1]), ord(out[1:2]), ord(out[2:3]), ord(out[3:4])

         elif input != '':
            ser.write(input + '\r\n')
            out = ''
            time.sleep(0.05)
            while ser.inWaiting() > 0:
               out += ser.read()
            if out != '':
               print out

   elif char == chr(60):
      movementr = open("movement.txt", "r")
      savedmovement = movementr.read()
      print 'starting playback from', movementr
      for fileLine in savedmovement.split('\r\n'):
         if fileLine[2:] != '':
            time.sleep(float(fileLine[2:]))
               #print fileLine[2:], fileLine[:1]
            if fileLine[:1] == '8':
               ser.write('a\r\n') #do whatever you want to do when reads key 8
               out = ''
               print out
            elif fileLine[:1] =='4':
               ser.write('b\r\n') #do whatever you want to do when reads key 4
               out = '' #can be removed if no serial output is to be received
               time.sleep(0.05) #can be removed if no serial output is to be received
               while ser.inWaiting() > 0: #can be removed if no serial output is to be received
                  out += ser.read() #can be removed if no serial output is to be received

               if out != '': #can be removed if no serial output is to be received
                  print out #can be removed if no serial output is to be received
      print 'end of playback'

   elif char == chr(56):
      #to do: stop current movement go to idle
      ser.write('a\r\n') #do whatever you want to do when you press key 8
      out = ''
      time.sleep(0.05)
      while ser.inWaiting() > 0:
         out += ser.read()
      print out
      print char,

      #print str(round(time.clock(), 2))
      currentclock = str(round(time.clock(), 2))

      runtimew = open("runtime.txt", "w")
      runtimew.write(str(round(time.clock(), 2)))

      runtimer = open("runtime.txt", "r")
      savedclock = runtimer.read()

      if savedclock == '': savedclock = '0.00'
      timebetweenkeys = float(currentclock) - float(savedclock)
      print timebetweenkeys #time between key presses
      
      runtimew = open("runtime.txt", "w")
      runtimew.write(str(round(time.clock(), 2)))
      
      movementa = open("movement.txt", "a")
      movementa.write(str(char))
      movementa = open("movement.txt", "a")
      movementa.write(' ')
      movementa = open("movement.txt", "a")
      movementa.write(str(timebetweenkeys))
      movementa = open("movement.txt", "a")
      movementa.write('\r\n')


   elif char == chr(52):
      #to do: stop current movement go to idle
      ser.write('a\r\n') #do whatever you want to do when you press key 4
      out = ''
      time.sleep(0.05)
      while ser.inWaiting() > 0:
         out += ser.read()
      print out
      print char,

      #print str(round(time.clock(), 2))
      currentclock = str(round(time.clock(), 2))

      runtimew = open("runtime.txt", "w")
      runtimew.write(str(round(time.clock(), 2)))

      runtimer = open("runtime.txt", "r")
      savedclock = runtimer.read()

      if savedclock == '': savedclock = '0.00'
      timebetweenkeys = float(currentclock) - float(savedclock)
      print timebetweenkeys #time between key presses
      
      runtimew = open("runtime.txt", "w")
      runtimew.write(str(round(time.clock(), 2)))
      
      movementa = open("movement.txt", "a")
      movementa.write(str(char))
      movementa = open("movement.txt", "a")
      movementa.write(' ')
      movementa = open("movement.txt", "a")
      movementa.write(str(timebetweenkeys))
      movementa = open("movement.txt", "a")
      movementa.write('\r\n')
