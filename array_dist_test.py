import scipy.spatial
import numpy

#a = [(0.0040668905644733301, 6.8121616554542766e-06, 4.2797474916872824e-10, 3.6353175568047615e-10, 1.3324856969120225e-19, 4.2145508014624666e-13, 5.2970414558065116e-20)]
#b = [(0.004501234828657873, 8.7202281851489338e-06, 8.5451640964827726e-09, 1.0360964596725404e-09, 5.7628729225461142e-19, 7.5989242209899085e-13, 3.0285658884835227e-18)]

#a = numpy.empty(1,2)
#b=numpy.array([[], []])
a=[]
a.append([2,3,4,5])
#a.extend([2,3,4,5])
b=[]
b.append([4,5,6,7])
#b.extend([4,5,6,7])

a.append([2,1,4,5])
#a.extend([4,5,6,7])

b.append([4,5,6,7])
#b.extend([4,5,6,7])

#a = numpy.array([[10,20,30],[40,50,60],[70,80,90]])
print a
#a = numpy.append(a,[[15,15,15]],axis=0)
print "-----------"
#print a[0]
#print a[0][0]

#aa = numpy.array(a)
#print "shape=", aa.shape

#anumpy.array([[0], [3,8,9,8]])
#b=numpy.array([[1,2,3,4], [3,8,9,8]])


print len(a)
print scipy.spatial.distance.cdist(a,b,'euclidean')
print scipy.spatial.distance.cdist(a,b, 'minkowski', 2)
print scipy.spatial.distance.cdist(a,b, 'seuclidean', V=None)
print scipy.spatial.distance.cdist(a,b,'correlation')
