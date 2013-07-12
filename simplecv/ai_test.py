

import os
import glob
from SimpleCV import *
print SimpleCV.__version__

"""
" Objective : experimenting object classification with SimpleCV
" Tasks : Comparing different generative and discriminative classifiers
"         Comparing different feature extractors
" Object Recognition : [1] Representation [2] Learning [3] Recognition
"""

print '******* Machine Learning Experiments ********'
print 'Choice of Extractors : '
print ' * bow - Bag of words Feature Extractor'
print ' * hue - Hue Histogram Feature Extractor'
print ' * morphology - Morphology Feature Extractor'
print ' * haar - Haar Like Feature Extractor'
print ' * edge - Edge Histogram Feature Extractor'
print 'Choice of Classifiers : '
print ' * naive - Naive Bayes Classifier'
print ' * svm - SVM Classifier'
print ' * knn - KNN Classifier'
print ' * tree - Tree Classifier'
print '******* Machine Learning Experiments ********'


# Training data set paths for classification(suppervised learnning)
image_dirs = ['../../coin_images/jheads/',
              '../../coin_images/jtails/',
			  '../../coin_images/oheads/',
              '../../coin_images/otails/',
              ]
# Different class labels for multi class classification
class_names = ['jhead','jtail','ohead', 'otail']

more = 'yes'
# [1] Representation - these feature extrators will extract features from image and represent it
feature_extractors = []
extractor_names = []

while more == 'yes':
    extractor_name = raw_input('Enter the feature extractor : ')
    if extractor_name == 'bow':
        feature_extractor = BOFFeatureExtractor() # feature extrator for bag of words methodology
        feature_extractor.generate(image_dirs,imgs_per_dir=40) # code book generation
    elif extractor_name == 'hue':
        feature_extractor = HueHistogramFeatureExtractor()
    elif extractor_name == 'morphology':
        feature_extractor = MorphologyFeatureExtractor()
    elif extractor_name == 'haar':
        feature_extractor = HaarLikeFeatureExtractor()
    elif extractor_name == 'edge':
        feature_extractor = EdgeHistogramFeatureExtractor()
    feature_extractors.append(feature_extractor)
    extractor_names.append(extractor_name)
    print "extractor_names:", extractor_names, feature_extractors
    more = raw_input('Do you want to continue : ')
       
# initializing classifier with appropriate feature extractors list
classifier_name = raw_input('Enter the classifier : ')
if classifier_name == 'naive':
    classifier = NaiveBayesClassifier(feature_extractors)
elif classifier_name == 'svm':
    classifier = SVMClassifier(feature_extractors)
elif classifier_name == 'knn':
    classifier = KNNClassifier(feature_extractors, 2)
elif classifier_name == 'tree':
    classifier = TreeClassifier(feature_extractors)
# train the classifier to generate hypothesis function for classification
print "image_dirs:", image_dirs, class_names
try:
    classifier.train(image_dirs,class_names,savedata='train.txt')
except:
	pass

classifier.save('ai.pkl')
classifier = None
classifier = NaiveBayesClassifier.load('ai.pkl')

raw_input('press enter to continue : ')
# Test data set paths
# here the 1-5 images are football images
# here the 6-10 images are cherries images
# here the 11-15 images are faces images
test_images_path = "../../coin_images/unclassified"
extension = "*.jpg"

if not test_images_path:
    path = os.getcwd() #get the current directory
else:
    path = test_images_path

directory = os.path.join(path, extension)
files = glob.glob(directory)

count = 0 # counting the total number of training images
error = 0 # conuting the total number of misclassification by the trained classifier
for image_file in files:
    new_image = Image(image_file)
    category = classifier.classify(new_image)
    if image_file[-9] == 'h':
        print "image_file:", image_file + "     classified as: " + category
        if category != 'jhead':
            print "INCORRECT CLASSIFICATION"
            error += 1
    if image_file[-9] == 't':
        print "image_file:", image_file + "     classified as: " + category
        if category != 'jtail':
            print "INCORRECT CLASSIFICATION"
            error += 1
    count += 1
# reporting the results
print ' * classifier : ' + classifier_name
print ' * extractors :', extractor_names
print ' *', error, 'errors out of', count
