#!/usr/bin/env python2
import nltk.metrics
from collections import defaultdict
import nltk.classify
from random import shuffle
from sklearn.svm import LinearSVC
	

def splitData(featuresets,testSize=100):
	posFeatureSets = [ (doc,cat) for (doc,cat) in featuresets if cat == 'pos'];
	negFeatureSets = [ (doc,cat) for (doc,cat) in featuresets if cat == 'neg'];

	shuffle(posFeatureSets);
	shuffle(negFeatureSets);

	train_set = posFeatureSets[testSize:] + negFeatureSets[testSize:]
	test_set = posFeatureSets[:testSize] + negFeatureSets[:testSize]

	return (train_set,test_set)



def getClassifier(train_set,type='NaiveBayes'):
	if type == 'NaiveBayes':
		return nltk.classify.NaiveBayesClassifier.train(train_set)
	elif type == 'SVM':
		classifier = nltk.classify.SklearnClassifier(LinearSVC())
		classifier.train(train_set)
		return classifier
	else:
		raise Exception('Unknown type')

def classify(classifier,test_set,echo=False):
	refsets = defaultdict(set)
	testsets = defaultdict(set)
	refLabels = list()
	observedLabels = list()

	#get orginal and predicted label of each testdata
	for i, (featureVector, label) in enumerate(test_set):  # i index
		observed = classifier.classify(featureVector)
		testsets[observed].add(i)
		refsets[label].add(i)
		refLabels.append(label)
		observedLabels.append(observed)

	posPrecision = nltk.metrics.precision(refsets['pos'], testsets['pos'])
	posRecall = nltk.metrics.recall(refsets['pos'], testsets['pos'])
	posFmeasure = nltk.metrics.f_measure(refsets['pos'], testsets['pos'])

	negPrecision = nltk.metrics.precision(refsets['neg'], testsets['neg'])
	negRecall  = nltk.metrics.recall(refsets['neg'], testsets['neg'])
	negFmeasure = nltk.metrics.f_measure(refsets['neg'], testsets['neg'])

	accuracy = nltk.metrics.accuracy(refLabels,observedLabels)

	if echo :
		print '  Postive  '
		print '-----------'
		print 'Precision:', posPrecision	
		print 'Recall   :', posRecall
		print 'F-measure:', posFmeasure
		print ''
		print ' Negitive '
		print '-----------'
		print 'Precision:', negPrecision
		print 'Recall   :', negRecall
		print 'F-measure:', negFmeasure
		print ''
		print 'Accuracy :', accuracy
		print ''
		classifier.show_most_informative_features()

	return (accuracy,posPrecision,posRecall,posFmeasure,
		negPrecision,negPrecision,negFmeasure)

def printTable(resultList):
	if len(resultList) == 0:
		return
	print ""
	print '|===============================================================================================|'
	print '|%3s | %10s | %10s   %10s   %10s | %10s   %10s   %10s |' % (
		'','Accuracy','','Postive','','','Negitive','')
	print '|%3s | %10s | %10s | %10s | %10s | %10s | %10s | %10s |' % (
		'','','Precision','Recall','F-measure','Precision','Recall','F-measure')
	
	print '|===============================================================================================|'

	for i,values in enumerate(resultList):
		print '|%3d |' % i,
		print '%10.8f | %10.8f | %10.8f | %10.8f | %10.8f | %10.8f | %10.8f |' % values

	print '|-----------------------------------------------------------------------------------------------|'
	avg = [sum(x)/len(resultList) for x in zip(*resultList)];
	print '|avg | %10.8f | %10.8f | %10.8f | %10.8f | %10.8f | %10.8f | %10.8f |' % tuple(avg)
	print '|===============================================================================================|'
	print ''
