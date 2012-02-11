import numpy as np
import mlpy
#from numpy import *
#from mlpy import *


aa = np.array([1,2,3,4])
bb = np.array([4,5,6,7])
cc = np.array([11,22,33,44])
dd = np.array([55,656,77,888])
#ee = 

xtr = np.array([ [cc,bb],  [aa,dd] ])

#xtr = np.zeros( (1,3,2) )

#print ee
#print "----------------"
print xtr
print 'np.size(xtr)', np.size(xtr) , 'np.shape(xtr)',np.shape(xtr), np.ndim(xtr), xtr.dtype



print "------------------------------------------"
mean1, cov1, n1 = [1, 5], [[1,1],[1,2]], 50  # 200 samples of class 1
#x1 = np.random.multivariate_normal(mean1, cov1, n1)
x1 = np.concatenate((mean1, cov1, n1), axis=0)
print x1.shape
print mean1, cov1, n1
print x1[0]
#ytr = np.array([1, 2])             # classes
#print ytr 
#print "np.size(ytr):", np.size(ytr), " np.shape(ytr):", np.shape(ytr)
#Save and read data from disk
#print mlpy.data_tofile('data_example.dat', xtr, ytr, sep='	')
#x, y = mlpy.data_fromfile('data_example.dat')
#print x
#print y

#print "mlpy.data_normalize(x) = ", mlpy.data_normalize(x)
