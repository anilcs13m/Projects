from classifier import splitData,getClassifier,printTable,classify
from corpusReader import PolarityDataReader


dataReader = None;

def loadData():
	global dataReader
	dataReader = PolarityDataReader()
	dataReader.getDocuments();
	dataReader.setTerms(2000,featureSelection=None)

def getData():
	featuresets = dataReader.getTermDocMatrix()
	return featuresets
	
def main():
	loadData();
	data = getData();

	NBresults = list()
	SVMresults = list()
	for x in xrange(0,10):
		(train_set,test_set) = splitData(data)
		
		classifier = getClassifier(train_set,'NaiveBayes')
		NBresults.append(classify(classifier,test_set))

		classifier = getClassifier(train_set,'SVM')
		SVMresults.append(classify(classifier,test_set))
	
	print '\nNaive Bayes Classifier'
	printTable(NBresults)

	print '\nSVM'
	printTable(SVMresults)

if __name__ == '__main__':
	main()
