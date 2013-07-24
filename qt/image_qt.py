#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sip
sip.setapi('QString', 2)
sip.setapi('QVariant', 2)

import time

from PyQt4 import QtGui, QtCore

class ImageChanger(QtGui.QWidget):    
    def __init__(self, images, parent=None):
        super(ImageChanger, self).__init__(parent)        

        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.addItems(images)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.comboBox)

class MyWindow(QtGui.QWidget):
    def __init__(self, images, parent=None):
        super(MyWindow, self).__init__(parent)
        self.label = QtGui.QLabel(self)

        self.index = 0
        print self.index 
        self.Picture1 =  QtGui.QPixmap(images[self.index ])

        #self.imageChanger = ImageChanger(images)
        #self.imageChanger.move(self.imageChanger.pos().y(), self.imageChanger.pos().x() + 100)
        #self.imageChanger.show()
        #self.imageChanger.comboBox.currentIndexChanged[str].connect(self.changeImage)
        btn = QtGui.QPushButton('Button', self)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        btn.resize(btn.sizeHint())
        btn.move(50, 50)       
        
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Tooltips')    
        self.show()

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.label)
        self.connect( btn, QtCore.SIGNAL( "clicked()" ),  self.setPicture1 )


    def setPicture1( self ):
            self.index = self.index + 1
            if self.index > (len(images)-1): self.index = 0
            #if self.index == 1: self.index = 2
            #if self.index ==2: self.index = 1  
            print self.index
            self.Picture1 =  QtGui.QPixmap(images[self.index ])
            self.label.setPixmap( self.Picture1 )

    @QtCore.pyqtSlot(str)
    def changeImage(self, pathToImage):
        pixmap = QtGui.QPixmap(pathToImage)
        self.label.setPixmap(pixmap)


if __name__ == "__main__":
    import sys

    images = [  "../../coin_images/otails/otail1.jpg",
                "../../coin_images/otails/otail2.jpg",
                ]

    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('MyWindow')

    main = MyWindow(images)
    main.show()

    sys.exit(app.exec_())
