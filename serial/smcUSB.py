import serial
import time

def log(*msgline):
    for msg in msgline:
        print msg,
    print

class smcUSB(object):
    #def __init__(self,con_port="COM6",ser_port="COM7",timeout=1): #/dev/ttyACM0  and   /dev/ttyACM1  for Linux
    def __init__(self,con_port="/dev/ttyACM0",ser_port="/dev/ttyACM1",timeout=1): #/dev/ttyACM0  and   /dev/ttyACM1  for Linux

        ############################
        # lets introduce and init the main variables
        self.con = None
        self.ser = None
        self.isInitialized = False
        
        ############################
        # lets connect the TTL Port
        try:
            self.con = serial.Serial(con_port,timeout=timeout)
            self.con.close()
            self.con.open()
            log("Link to Command Port -", con_port, "- successful")

        except serial.serialutil.SerialException, e:
            print e
            log("Link to Command Port -", con_port, "- failed")

        if self.con:
            #####################
            #If your Maestro's serial mode is "UART, detect baud rate", you must first send it the baud rate indication byte 0xAA on
            #the RX line before sending any commands. The 0xAA baud rate indication byte can be the first byte of a Pololu protocol
            #command.
            #http://www.pololu.com/docs/pdf/0J40/maestro.pdf - page 35
            self.con.write(chr(0xAA))
            self.con.flush()
            log("Baud rate indication byte 0xAA sent!")
        
        ###################################
        # lets connect the TTL Port
        try:
            self.ser = serial.Serial(ser_port,timeout=timeout)
            self.ser.close()
            self.ser.open()
            log("Link to TTL Port -", ser_port, "- successful")
        except serial.serialutil.SerialException, e:
            print e
            log("Link to TTL Port -", ser_port, "- failed!")
        
        self.isInitialized = (self.con!=None and self.ser!=None)
        if (self.isInitialized):
            err_flags = self.get_errors()
            log("Device error flags read (",err_flags,") and cleared")
        log("Device initialized:",self.isInitialized)

    ###########################################################################################################################
    ## common write function for handling all write related tasks
    def write(self,*data):
        if not self.isInitialized: log("Not initialized"); return
        #if not self.ser.writable():
        #    log("Device not writable")
        #    return
        for d in data:
            self.ser.write(chr(d))
        self.ser.flush()

