from mm18usb import *

class CoinServoDriver(object):
  def __init__(self,x_servo=0,y_servo=1,z_servo=2):
    self.x_servo = x_servo
    self.y_servo = y_servo
    self.z_servo = z_servo
    self.device = mm18usb()
    self.device.set_acceleration(self.x_servo,10)
    self.device.set_speed(self.x_servo,10)
    self.device.set_acceleration(self.y_servo,10)
    self.device.set_speed(self.y_servo,10)
    self.device.set_acceleration(self.z_servo,10)
    self.device.set_speed(self.z_servo,10)
    self.device.go_home()

  def __del__(self):
    del(self.device)
    
  def status_report(self):
    return "X: %s\tY: %s\tZ: %s" % (self.device.get_position(self.x_servo),self.device.get_position(self.y_servo),self.device.get_position(self.z_servo))

  def pan(self,dx):
    x = self.device.get_position(self.x_servo)
    x += dx
    self.device.set_target(self.x_servo,x)
    self.device.wait_until_at_target()
 
  def tilt(self,dy):
    y = self.device.get_position(self.y_servo)
    y += dy
    self.device.set_target(self.y_servo,y)
    self.device.wait_until_at_target()

  def rotate(self,dz):
    z = self.device.get_position(self.z_servo)
    z += dz
    self.device.set_target(self.z_servo,z)
    self.device.wait_until_at_target()
    
  def goto(self,x,y,z=0):
    self.device.set_target(self.x_servo,x)
    self.device.set_target(self.y_servo,y)
    self.device.set_target(self.z_servo,z)
    self.device.wait_until_at_target()
    
  def reset(self):
    self.device.go_home()
    self.device.wait_until_at_target()

