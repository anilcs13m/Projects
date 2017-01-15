import os
import re
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score as AUC

######################################################################	
#    The first file that you'll need is "unlabeledTrainData.tsv",    #
#    which contains 25,000 IMDB movie reviews,                       #
#    each with a positive or negative sentiment label.               #
######################################################################

#################################################################################################
# Import the pandas package, then use the "read_csv" function to read the labeled training data #
#################################################################################################
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
###############################################################################################
# Here, "header=0" indicates that the first line of the file contains column names,           #
# "delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 tells         #
# Python to ignore doubled quotes, otherwise you may encounter errors trying to read the file.#
###############################################################################################
print("just checking shape and columns names of training data")
print(train.shape)
print(train.columns.values)
#####################################################################################
# Second file "testData.tsv" This file contains another 25,000 reviews and ids;     #
# our task is to predict the sentiment label.                                       #
#####################################################################################

# Import the pandas package, then use the "read_csv" function to read the test data	 
test = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
################################################################################################
#  Here, "header=0" indicates that the first line of the file contains column names,           #
# "delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 tells          #
#  Python to ignore doubled quotes, otherwise you may encounter errors trying to read the file.#
################################################################################################
print("just checking shape and columns names of test data")
print(test.shape)
print(test.columns.values)

unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0,  delimiter="\t", quoting=3 )

# just viewing some of the text file
print('The first review is:')
print(train["review"][0])


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

# some preprocessing and text data cleaning
def review_to_wordlist( review, remove_stopwords=False ):
	"""
        Function to convert a document to a sequence of words,
        optionally removing stop words.  Returns a list of words.
    """
        # 1. Text data may contain some HTML tags so Remove HTML
        # beautifulsoup is use for cleaning HTNL tags from text file
        review_text = BeautifulSoup(review).get_text()
        
        # 2. some times non letter does't carry any meaning in text analysis, remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)

        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        
        # 4. Some does't contain meaningfull information we useally called such words as 
        #    stop words remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
       # return words list
        return(words)

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        """
         	Function to split a review into parsed sentences. Returns a
        	list of sentences, where each sentence is a list of words
        """	
        #1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append(review_to_wordlist(raw_sentence, remove_stopwords ))
    	# return sentance
        return sentences

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews


# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# ****** Split the labeled and unlabeled training sets into clean sentences
#
sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

# ****** Set parameters and train the word2vec model
#
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training Word2Vec model..."
model = Word2Vec(sentences, workers=num_workers, size=num_features, 
                            min_count = min_word_count,  
                            window = context, 
                            sample = downsampling, seed=1)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# ****** Create average vectors for the training and test sets
print "Creating average feature vecs for training reviews"

trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features )

print "Creating average feature vecs for test reviews"

testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )

# ****** Fit a random forest to the training set, then make predictions
# Fit a random forest to the training data, using 100 trees
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainDataVecs, train["sentiment"] )

# Use the random forest to make sentiment label predictions
print "Predicting test labels...\n"
rf_p = forest.predict_proba(testDataVecs)
auc = AUC( test['sentiment'].values, rf_p[:,1] )
print "random forest AUC:", auc
# a random score from a _random_ forest
# let's define a helper function
def train_and_eval_auc( model, train_x, train_y, test_x, test_y ):
    model.fit( train_x, train_y )
    p = model.predict_proba(test_x )
    auc = AUC( test_y, p[:,1] )
    return auc

# train a random forest ten times, average the scores
rf_aucs = []
for i in range( 10 ):
    auc = train_and_eval_auc(forest, trainDataVecs, train["sentiment"],
                              testDataVecs, test["sentiment"].values
                              print "random forest run {}, AUC: {}".format( i, auc )
    rf_aucs.append( auc )

avg_auc = sum( rf_aucs ) / len( rf_aucs )
print "Average AUC from random forest:", avg_auc
