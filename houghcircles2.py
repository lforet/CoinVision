import cv


#cv.NamedWindow("camera", 1)
capture = cv.CaptureFromCAM(0)

while True:
    img = cv.QueryFrame(capture)
    gray = cv.CreateImage(cv.GetSize(img), 8, 1)
    edges = cv.CreateImage(cv.GetSize(img), 8, 1)

    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
    cv.Canny(gray, edges, 50, 200, 3)
    cv.Smooth(gray, gray, cv.CV_GAUSSIAN, 9, 9)

    storage = cv.CreateMat(1, 2, cv.CV_32FC3)

    print storage[0,0]
    #This is the line that throws the error
    print cv.HoughCircles(gray, storage, cv.CV_HOUGH_GRADIENT, 2, gray.height/4, 200, 100)
    cv.ShowImage("camera", gray)
    if cv.WaitKey(10) == 1048603:
	break


