import os
import re
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

##################################################################
# The first file that you'll need is "unlabeledTrainData.tsv",   #
# which contains 25,000 IMDB movie reviews, each with a positive #
# or negative sentiment label.                                   #
##################################################################

################################################################################################
#Import the pandas package, then use the "read_csv" function to read the labeled training data #
################################################################################################
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
################################################################################################
# Here, "header=0" indicates that the first line of the file contains column names,            #
# "delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 tells          #
# Python to ignore doubled quotes, otherwise you may encounter errors trying to read the file. #
################################################################################################
print("just checking shape and columns names of training data")
print(train.shape)
print(train.columns.values)
#################################################################################
# Second file "testData.tsv" This file contains another 25,000 reviews and ids; #
# our task is to predict the sentiment label.                                   #
#################################################################################

#####################################################################################
# Import the pandas package, then use the "read_csv" function to read the test data	#
##################################################################################### 
test = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
#################################################################################################
# Here, "header=0" indicates that the first line of the file contains column names,             #
# "delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 tells           #
# Python to ignore doubled quotes, otherwise you may encounter errors trying to read the file.  #
#################################################################################################
print("just checking shape and columns names of test data")
print(test.shape)
print(test.columns.values)

# just viewing some of the text file
print('The first review is:')
print(train["review"][0])

# some preprocessing and text data cleaning
def review_to_wordlist( review, remove_stopwords=False ):
	#################################################################
    #   Function to convert a document to a sequence of words,      #
    #   optionally removing stop words.  Returns a list of words.   #  
    #   1. Text data may contain some HTML tags so Remove HTML      #
    #   beautifulsoup is use for cleaning HTNL tags from text file  #
    #################################################################
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
        #################################################################
        # 	Function to split a review into parsed sentences. Returns a #
        #	list of sentences, where each sentence is a list of words   #
        #################################################################

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

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

print "Cleaning and parsing the training set movie reviews...\n"
for i in xrange( 0, len(train["review"])):
	clean_train_reviews.append(" ".join(review_to_wordlist(train["review"][i], True)))

# Create an empty list and append the clean reviews one by one
clean_test_reviews = []

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,len(test["review"])):
	clean_test_reviews.append(" ".join(review_to_wordlist(test["review"][i], True)))


print("**********Creating the bag of words of reviews**************\n")
#####################################################################
# creation of bag of words can be done by using "CountVectorizer"   #
# Initialize the "CountVectorizer" object, which is scikit-learn's  #
# bag of words tool.                                                #
#####################################################################

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
#####################################################################
# fit_transform() does two functions:                               #
# 1. First, it fits the model and learns the vocabulary;            #
# 2. second, it transforms our training data into feature vectors.  #
# The input to fit_transform should be a list of strings.           #
#####################################################################

#########################################################################
# Numpy arrays are easy to work with, so convert the result to an array #
#########################################################################
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

#####################################################################
# Get a bag of words for the test set, and convert to a numpy array #
#####################################################################
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


grid_values = {'C':[30]} # Decide which settings you want for the grid search. 

model_LR = GridSearchCV(LogisticRegression(penalty = 'L2', dual = True, random_state = 0), grid_values, scoring = 'roc_auc', cv = 20) 
###########################################################################
# Try to set the scoring on what the contest is asking for.               #
# The contest says scoring is for area under the ROC curve, so use this.  #
###########################################################################

model_LR.fit(train_data_features, train["sentiment"] )

# Use the random forest to make sentiment label predictions
print "Predicting test labels...\n"
result = model_LR.predict(test_data_features)

print(model_LR.grid_scores_)
print(model_LR.best_estimator_)