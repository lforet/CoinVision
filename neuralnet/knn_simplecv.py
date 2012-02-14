
from SimpleCV import *
import sys


#inp = raw_input("Do you want to continue [Y/n]")
#if not (inp == "" or inp.lower() == "y"):
#    print "Exiting the program"
#    sys.exit()

#machine_learning_data_set = "https://github.com/downloads/ingenuitas/SimpleCV/machine_learning_dataset.zip"
#data_path = download_and_extract(machine_learning_data_set)
data_path = "/tmp/tmpXvh8GJ"
print data_path 


spath = data_path + "/data/structured/"
upath = data_path + "/data/unstructured/"
ball_path = spath+"ball/"
basket_path = spath+"basket/"
boat_path = spath+"boat/"
cactus_path = spath +"cactus/"
cup_path = spath+"cup/"
duck_path = spath+"duck/"
gb_path = spath+"greenblock/"
match_path = spath+"matches/"
rb_path = spath+"redblock/"
s1_path = spath+"stuffed/"
s2_path = spath+"stuffed2/"
s3_path = spath+"stuffed3/"

arbor_path = upath+"arborgreens/"
football_path = upath+"football/"
sanjuan_path = upath+"sanjuans/"

w = 800
h = 600
n=-1

display = Display(resolution = (w,h))

morph = MorphologyFeatureExtractor()
hue = HueHistogramFeatureExtractor(mNBins=16)
edge = EdgeHistogramFeatureExtractor(bins=16)



i = 0
#while not display.isDone():
while display.isNotDone():
	if display.mouseLeft:
		print "i = ", i
		if i < 8 :i = i + 1
		if i == 8: break
	
	if i == 1:
		print ('##########################################################################')
		print('Train')
		
		# now try an RBF kernel
		#extractors = [hue,edge]
		extractors = [edge]
		path = [cactus_path,cup_path,basket_path]
		classes = ['cactus','cup','basket']
		#path = [s1_path,s2_path,s3_path]
		#classes = ['s1','s2','s3']
		
		props ={
				'KernelType':'RBF', #default is a RBF Kernel
				'SVMType':'NU', #default is C
				'nu':None, # NU for SVM NU
				'c':None, #C for SVM C - the slack variable
				'degree':None, #degree for poly kernels - defaults to 3
				'coef':None, #coef for Poly/Sigmoid defaults to 0
				'gamma':None, #kernel param for poly/rbf/sigma
			}

		"""
		print('SVMRBF ')
		classifierSVMRBF = SVMClassifier(extractors,props)
		classifierSVMRBF.train(path,classes,disp=display,subset=n) #train
		
		##############################
		#print('Forest')
		#extractors = [morph]
		#classifierForest = TreeClassifier(extractors,flavor='Forest')#
		#classifierForest.train(path,classes,disp=display,subset=n) #train
		################################
		print "KNN"
		classifierKNN = KNNClassifier(extractors)#
		classifierKNN.train(path,classes,disp=display,subset=n) #train
		print "Done Training"
		i = 2
		

	if i == 3:	
		print('Test')
		[pos,neg,confuse] = classifierSVMRBF.test(path,classes,disp=display,subset=n)
		#[pos,neg,confuse] = classifierForest.test(path,classes,disp=display,subset=n)
		[pos,neg,confuse] = classifierKNN.test(path,classes,disp=display,subset=n)
		#files = glob.glob( os.path.join(path[0], '*.jpg'))
		#for i in range(10):
		#		img = Image(files[i])
		#		cname = classifierSVMRBF.classify(img)
		#		print(files[i]+' -> '+cname)
		classifierSVMRBF.save('RBFSVM.pkl')
		#classifierForest.save('forest.pkl')
		classifierKNN.save('knn.pkl')
		print "done testing"
		i = 4
	"""
	if i == 5:
		print('Reloading from file')
		extractors = [edge]
		testSVMRBF = SVMClassifier.load('RBFSVM.pkl')
		testSVMRBF.setFeatureExtractors(extractors)
		#testForest = TreeClassifier.load('forest.pkl')
		#extractors = [morph]
		#testForest.setFeatureExtractors(extractors)
		testKNN = KNNClassifier.load('knn.pkl')
		testKNN.setFeatureExtractors(extractors)
		files = glob.glob( os.path.join(path[1], '*.jpg'))
		for i in range(len(files)):
				img = Image(files[i])
				cname = testSVMRBF.classify(img)
				print("SVM ", files[i]+' -> '+cname)
				#cname = testForest.classify(img)
				#print(files[i]+' -> '+cname)
				cname = testKNN.classify(img)
				print("knn ", files[i]+' -> '+cname)


		print "completed..."
		i = 6
