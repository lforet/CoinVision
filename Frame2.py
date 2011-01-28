#Boa:Frame:Frame2

import wx
import os
import cv
import pygame
import time


def create(parent):
    return Frame2(parent)

imgs = {'.bmp' : wx.BITMAP_TYPE_BMP,
        '.gif' : wx.BITMAP_TYPE_GIF,
        '.png' : wx.BITMAP_TYPE_PNG,
        '.jpg' : wx.BITMAP_TYPE_JPEG,
        '.ico' : wx.BITMAP_TYPE_ICO}

capture = 0

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

def gray_images(img):

	temp_img = cv.CreateImage(cv.GetSize(img), 8, 1)
	if img.channels == 1:
		temp_img = img
	if img.channels > 1:
		cv.CvtColor(img, temp_img, cv.CV_BGR2GRAY)
	return(temp_img)

def get_coin_center(img):

	temp = cv.CloneImage(img)
	gray = cv.CreateImage(cv.GetSize(temp), 8, 1)
	print temp.channels, temp.depth, gray.channels	
	print type(temp), type(gray)
	if img.channels != 1: cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
	best_circle = (0,0,0)
	#print best_circle 
	#cv.Smooth(edges, edges, cv.CV_GAUSSIAN, 9, 9)
	for i in range (140, 225):
		#print i
		storage = cv.CreateMat(50, 1, cv.CV_32FC3)
		cv.SetZero(storage)
		cv.HoughCircles(gray, storage, cv.CV_HOUGH_GRADIENT, 1, float(40), float(175), float(55), long(i),long(230))

		num_of_circles = storage.rows
		
		for ii in range(num_of_circles):
			circle_data = storage[ii,0]
			center = cv.Round(circle_data[0]), cv.Round(circle_data[1])
			radius = cv.Round(circle_data[2])
			#print circle_data[0], circle_data[1], circle_data[2]
			if radius > 140:
				if radius > best_circle[2]:  
					#print "best was = ", best_circle
					best_circle = (circle_data[0], circle_data[1], circle_data[2])
					#print "best now = ", i				

 	return (best_circle)

def scale_and_crop(img1, img2):
	size_buffer = 15
	radius_buffer = 50
	coin1 = get_coin_center(img1)
	coin2 = get_coin_center(img2)
	print "coin1, coin2 = ", coin1, coin2 
	print coin1[2]-coin2[2]
	 
	coin1_center = int(coin1[0]), int(coin1[1])
	coin1_radius = int(coin1[2])
	coin1_inside_radius = coin1_radius - radius_buffer
	coin2_center = int(coin2[0]), int(coin2[1])
	coin2_radius = int(coin2[2])
	coin2_inside_radius = coin2_radius - radius_buffer

	#crop OUTSIDE bounding rectangle for orientation 
	#topleft_corner1 = (coin1_center[0]-coin1_radius-size_buffer, coin1_center[1]-coin1_radius-size_buffer)
	#bottomright_corner1 = (coin1_center[0]+coin1_radius+size_buffer, coin1_center[1]+coin1_radius+size_buffer)
	#topleft_corner2 = (coin2_center[0]-coin2_radius-size_buffer, coin2_center[1]-coin2_radius-size_buffer)
	#bottomright_corner2 = (coin2_center[0]+coin2_radius+size_buffer, coin2_center[1]+coin2_radius+size_buffer)
	#crop inside bounding rectangle for orientation 
	topleft_corner1 = (coin1_center[0]-int((coin1_inside_radius*(cv.Sqrt(2)/2))), coin1_center[1]-int((coin1_inside_radius*(cv.Sqrt(2)/2))))
	bottomright_corner1 = (coin1_center[0]+int((coin1_inside_radius*(cv.Sqrt(2)/2))), coin1_center[1]+int((coin1_inside_radius*(cv.Sqrt(2)/2))))
	topleft_corner2 = (coin2_center[0]-int((coin2_inside_radius*(cv.Sqrt(2)/2))), coin2_center[1]-int((coin2_inside_radius*(cv.Sqrt(2)/2))))
	bottomright_corner2 = (coin2_center[0]+int((coin2_inside_radius*(cv.Sqrt(2)/2))), coin2_center[1]+int((coin2_inside_radius*(cv.Sqrt(2)/2))))

	cropped_img1 = cv.GetSubRect(img1, (topleft_corner1[0], topleft_corner1[1], bottomright_corner1[0]-topleft_corner1[0], bottomright_corner1[1]-topleft_corner1[1]))
	cropped_img2 = cv.GetSubRect(img2, (topleft_corner2[0], topleft_corner2[1], bottomright_corner2[0]-topleft_corner2[0], bottomright_corner2[1]-topleft_corner2[1]))

	print "Before resize SIZES = ", cv.GetSize(cropped_img1), cv.GetSize(cropped_img2)
	temp_img = cv.CreateImage(cv.GetSize(cropped_img1), 8, img2.channels)
	temp_img2 = cv.CreateImage(cv.GetSize(cropped_img1), 8, img1.channels)
	cv.Resize(cropped_img2, temp_img)
	cv.Resize(cropped_img1, temp_img2)
	print "Before resize SIZES = ", cv.GetSize(cropped_img1), cv.GetSize(temp_img)
	#cv.WaitKey()
	return(temp_img2, temp_img)


def get_orientation(img1, img2):

	subtracted_image = cv.CreateImage(cv.GetSize(img1), 8, 1)
	temp_img = cv.CreateImage(cv.GetSize(img1), 8, 1)

	cv.Smooth(img1, img1, cv.CV_GAUSSIAN, 9, 9)
	cv.Smooth(img2, img2, cv.CV_GAUSSIAN, 9, 9)
	cv.Canny(img1,img1 ,87,175, 3)
	cv.Canny(img2,img2, 87,175, 3)
	cv.ShowImage("img1", img1)
	cv.ShowImage("img2", img2)
	cv.WaitKey()
	best_sum = 0
	best_orientation = 0
	for i in range(1, 360):
		temp_img = rotate_image(img2, i)
		cv.And(img1, temp_img , subtracted_image)
		cv.ShowImage("subtracted_image", subtracted_image)
		cv.ShowImage("Image of Interest", temp_img )
		sum_of_and = cv.Sum(subtracted_image)
		if best_sum == 0: best_sum = sum_of_and[0]
		if sum_of_and[0] > best_sum: 
			best_sum = sum_of_and[0]
			best_orientation = i
		print i, "Sum = ", sum_of_and[0], "  best_sum= ", best_sum , "  best_orientation =", best_orientation
		key = cv.WaitKey(5)
		if key == 27 or key == ord('q') or key == 1048688 or key == 1048603:
			break
		#time.sleep(.01)
	return (best_orientation)

def draw_boundries(img):

	size_buffer = 15
	radius_buffer = 50
	coin_center = get_coin_center(img)
	center = int(coin_center[0]), int(coin_center[1])
	radius = int(cv.Round(coin_center[2]))
	inside_radius = radius - radius_buffer
	print coin_center 
	temp = cv.CloneImage(img)
	cv.Circle(temp, (center), radius, cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0 )
	cv.Circle(temp ,(center), 2, cv.CV_RGB(0, 0, 255), 2, cv.CV_AA, 0 ) 	
	cv.Circle(temp ,(center), (radius - radius_buffer), cv.CV_RGB(0, 0, 255), 2, cv.CV_AA, 0 )

	#Draw outside bounding rectangle 
	topleft_corner = (center[0]-radius-size_buffer, center[1]-radius-size_buffer)
	bottomright_corner = (center[0]+radius+size_buffer, center[1]+radius+size_buffer)
	cv.Rectangle(temp, topleft_corner, bottomright_corner, cv.CV_RGB(255, 255, 0), 2, 0)
	#Draw inside bounding rectangle 
	topleft_corner = (center[0]-int((inside_radius*(cv.Sqrt(2)/2))), center[1]-int((inside_radius*(cv.Sqrt(2)/2))))
	bottomright_corner = (center[0]+int((inside_radius*(cv.Sqrt(2)/2))), center[1]+int((inside_radius*(cv.Sqrt(2)/2))))
	cv.Rectangle(temp, topleft_corner, bottomright_corner, cv.CV_RGB(255, 255, 0), 2, 0)
	return(temp)

[wxID_FRAME2, wxID_FRAME2BUTTON1, wxID_FRAME2BUTTON2, wxID_FRAME2PANEL1, 
 wxID_FRAME2PANEL2, wxID_FRAME2PANEL3, wxID_FRAME2SASHWINDOW1, 
 wxID_FRAME2SASHWINDOW2, wxID_FRAME2STATICBITMAP1, wxID_FRAME2STATICBITMAP2, 
 wxID_FRAME2STATUSBAR1, 
] = [wx.NewId() for _init_ctrls in range(11)]

[wxID_FRAME2MENU1ITEMS0, wxID_FRAME2MENU1ITEMS1, 
] = [wx.NewId() for _init_coll_menu1_Items in range(2)]

class Frame2(wx.Frame):
    def _init_coll_boxSizer3_Items(self, parent):
        # generated method, don't edit

        parent.AddSizer(self.flexGridSizer2, 0, border=0, flag=0)

    def _init_coll_menuBar1_Menus(self, parent):
        # generated method, don't edit

        parent.Append(menu=self.menu1, title=u'Menus0')

    def _init_coll_menu1_Items(self, parent):
        # generated method, don't edit

        parent.Append(help='', id=wxID_FRAME2MENU1ITEMS0, kind=wx.ITEM_NORMAL,
              text='Items0')
        parent.AppendSeparator()
        parent.Append(help='', id=wxID_FRAME2MENU1ITEMS1, kind=wx.ITEM_NORMAL,
              text='Items1')
        self.Bind(wx.EVT_MENU, self.OnMenu1Items0Menu,
              id=wxID_FRAME2MENU1ITEMS0)

    def _init_sizers(self):
        # generated method, don't edit
        self.boxSizer3 = wx.BoxSizer(orient=wx.VERTICAL)

        self.flexGridSizer1 = wx.FlexGridSizer(cols=0, hgap=0, rows=1, vgap=0)

        self.flexGridSizer2 = wx.FlexGridSizer(cols=0, hgap=0, rows=1, vgap=0)

        self._init_coll_boxSizer3_Items(self.boxSizer3)

        self.button1.SetSizer(self.flexGridSizer1)
        self.panel3.SetSizer(self.boxSizer3)

    def _init_utils(self):
        # generated method, don't edit
        self.menu1 = wx.Menu(title=u'')

        self.menuBar1 = wx.MenuBar()

        self._init_coll_menu1_Items(self.menu1)
        self._init_coll_menuBar1_Menus(self.menuBar1)

    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_FRAME2, name='', parent=prnt,
              pos=wx.Point(786, 109), size=wx.Size(858, 858),
              style=wx.DEFAULT_FRAME_STYLE, title='Frame2')
        self._init_utils()
        self.SetClientSize(wx.Size(858, 858))
        self.SetMenuBar(self.menuBar1)

        self.statusBar1 = wx.StatusBar(id=wxID_FRAME2STATUSBAR1,
              name='statusBar1', parent=self, style=0)
        self.statusBar1.SetStatusText(u'')
        self.SetStatusBar(self.statusBar1)

        self.panel3 = wx.Panel(id=wxID_FRAME2PANEL3, name='panel3', parent=self,
              pos=wx.Point(0, 0), size=wx.Size(858, 804),
              style=wx.TAB_TRAVERSAL)

        self.panel1 = wx.Panel(id=wxID_FRAME2PANEL1, name='panel1',
              parent=self.panel3, pos=wx.Point(744, 24), size=wx.Size(96, 40),
              style=wx.TAB_TRAVERSAL)

        self.button1 = wx.Button(id=wxID_FRAME2BUTTON1, label=u'SnapShot',
              name='button1', parent=self.panel1, pos=wx.Point(0, 0),
              size=wx.Size(85, 29), style=0)
        self.button1.Bind(wx.EVT_BUTTON, self.OnButton1Button,
              id=wxID_FRAME2BUTTON1)

        self.panel2 = wx.Panel(id=wxID_FRAME2PANEL2, name='panel2',
              parent=self.panel3, pos=wx.Point(744, 64), size=wx.Size(104, 40),
              style=wx.TAB_TRAVERSAL)

        self.button2 = wx.Button(id=wxID_FRAME2BUTTON2, label=u'FindCenter',
              name='button2', parent=self.panel2, pos=wx.Point(0, 0),
              size=wx.Size(85, 29), style=0)
        self.button2.Bind(wx.EVT_BUTTON, self.OnButton2Button,
              id=wxID_FRAME2BUTTON2)

        self.sashWindow1 = wx.SashWindow(id=wxID_FRAME2SASHWINDOW1,
              name='sashWindow1', parent=self.panel3, pos=wx.Point(32, 24),
              size=wx.Size(640, 480), style=wx.CLIP_CHILDREN | wx.SW_3D)
        self.sashWindow1.SetExtraBorderSize(10)
        self.sashWindow1.SetDefaultBorderSize(10)
        self.sashWindow1.SetBestFittingSize(wx.Size(640, 480))

        self.staticBitmap1 = wx.StaticBitmap(bitmap=wx.NullBitmap,
              id=wxID_FRAME2STATICBITMAP1, name='staticBitmap1',
              parent=self.sashWindow1, pos=wx.Point(0, 0), size=wx.Size(640,
              480), style=0)
        self.staticBitmap1.Center(wx.BOTH)
        self.staticBitmap1.SetBestFittingSize(wx.Size(640, 480))

        self.sashWindow2 = wx.SashWindow(id=wxID_FRAME2SASHWINDOW2,
              name='sashWindow2', parent=self.panel3, pos=wx.Point(32, 530),
              size=wx.Size(320, 240), style=wx.CLIP_CHILDREN | wx.SW_3D)
        self.sashWindow2.SetBestFittingSize(wx.Size(320, 240))

        self.staticBitmap2 = wx.StaticBitmap(bitmap=wx.NullBitmap,
              id=wxID_FRAME2STATICBITMAP2, name='staticBitmap2',
              parent=self.sashWindow2, pos=wx.Point(0, 0), size=wx.Size(320,
              240), style=0)

        self._init_sizers()

    def __init__(self, parent):
        self._init_ctrls(parent)

    def OnMenu1Items0Menu(self, event):
        global base_img
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "*.*", wx.OPEN)
        try:
            if dlg.ShowModal() == wx.ID_OK:
                filename = dlg.GetPath()
                string = str(filename)
                self.statusBar1.SetStatusText(string)
                ext = os.path.splitext(filename)[-1].lower()
                #load image, convert to 1/2 size, resave then load in as a bmp for wxPython
                temp_img = cv.LoadImageM(filename)
                base_img = temp_img
                halfsize_temp_img = cv.CreateMat(temp_img.rows / 2, temp_img.cols / 2, cv.CV_8UC3)
                cv.Resize(temp_img, halfsize_temp_img)
                cv.SaveImage('halfsize_temp_img.png',halfsize_temp_img)
                
                bmp = wx.Image('halfsize_temp_img.png', imgs[ext]).ConvertToBitmap()
                print filename, ext
                #bmp = wx.Image("snapshot.png", wx.BITMAP_TYPE_PNG).ConvertToBitmap()
                self.sashWindow2.SetClientSize(wx.Size(bmp.GetWidth()+self.sashWindow2.ExtraBorderSize*2,
                                                       bmp.GetHeight()+self.sashWindow2.ExtraBorderSize*2))
                self.staticBitmap2.SetBitmap(bmp)
                string = str(bmp.GetWidth()) + "," + str(bmp.GetHeight())
                self.statusBar1.SetStatusText(string)
                #cv.ShowImage("ggg", base_img)
        finally:
            dlg.Destroy()
        event.Skip() 
        
    #def UpdateBaseImage(self, img):
    #    base_temp_img = cv.CreateMat(img.rows / 2, img.cols / 2, cv.CV_8UC3)
    #    cv.Resize(temp_img, halfsize_temp_img)
    #    cv.SaveImage('base_temp_img.png',img)    
    #    bmp = wx.Image('base_temp_img.png', imgs[ext]).ConvertToBitmap()
    #    self.sashWindow2.SetClientSize(wx.Size(bmp.GetWidth()+self.sashWindow2.ExtraBorderSize*2,
    #                                           bmp.GetHeight()+self.sashWindow2.ExtraBorderSize*2))
    #    self.staticBitmap2.SetBitmap(bmp)
    #    string = str(bmp.GetWidth()) + "," + str(bmp.GetHeight())
    #    self.statusBar1.SetStatusText(string)
       
        
    def OnButton1Button(self, event):
        global snap_img
        
        # capture the current frame
        capture = cv.CaptureFromCAM(0)
        frame = cv.QueryFrame(capture)
        snap_img = cv.CloneImage(frame)
        #cv.ShowImage('snap_img', snap_img)
        cv.SaveImage("snapshot.png", frame)
        #cv.ReleaseCapture(capture)
        #os.path = "~/projects/CoinVision/images/"
        #fn = "2011-01-20-154713.jpg"
        #path = os.path + fn
        #self.statusBar1.SetStatusText(path)
        #ext = os.path.splitext(fn)[-1].lower()
        #bmp = wx.Image("/home/lforet/projects/CoinVision/images/2011-01-20-154713.jpg", wx.BITMAP_TYPE_JPEG).ConvertToBitmap()
        self.UpdateSnapshot(self)

    def OnButton2Button(self, event):
        global base_img
        global snap_img
        base_img = cv.GetImage(base_img)
        img1_gray = cv.CloneImage(base_img)
        img2_gray = cv.CloneImage(snap_img)
        
        img1_gray = gray_images(img1_gray)
        img2_gray = gray_images(img2_gray)
        print get_orientation(img1_gray, img2_gray)
        #img1_gray, img2_gray = scale_and_crop(img1_gray, img2_gray)
        cv.SaveImage("snapshot.png", img2_gray)
        #pygame.mixer.music.load('./sounds/sound1.ogg')
        #pygame.mixer.music.play()
        #is_playing = pygame.mixer.music.get_busy()
        #while is_playing:
        #    time.sleep(1)
        #    is_playing = pygame.mixer.music.get_busy()
        #    print "busy= ", is_playing
            
        #pygame.event.wait()
        self.UpdateSnapshot(self)
        #bounded_coin_img1 = draw_boundries(base_img)
        #cv.ShowImage('bounded_coin_img1', bounded_coin_img1)
        #event.Skip()
   
    def UpdateSnapshot(self, event):
        bmp = wx.Image("snapshot.png", wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.sashWindow1.SetClientSize(wx.Size(bmp.GetWidth()+self.sashWindow1.ExtraBorderSize*2,
                                               bmp.GetHeight()+self.sashWindow1.ExtraBorderSize*2))
        #self.SetClientSize(self.sashWindow1.GetSize())
        self.staticBitmap1.SetBitmap(bmp)
        string = str(bmp.GetWidth()) + "," + str(bmp.GetHeight())
        self.statusBar1.SetStatusText(string)
    
if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = create(None)
    frame.Show()
    pygame.init()

    app.MainLoop()
