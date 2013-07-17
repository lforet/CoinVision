from random import choice
import sys
import itertools
import numpy as np
system_tried = []
classifiers = ['svm', 'knn', 'lr', 'naive', 'tree']
features = ['lbp', 'tas', 'hu']

ff = []
for i in range(1, len(features) +1):
	ff.extend((list(itertools.combinations(features, r=i))))

print ff
for i in range(0, len(ff)):
	print ff[i]

cl = []
for i in range(1, len(classifiers) +1):
	cl.extend((list(itertools.combinations(classifiers, r=i))))

print cl
for i in range(0, len(cl)):
	print cl[i]
sys.exit(-1)


