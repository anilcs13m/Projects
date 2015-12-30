from __future__ import division

import config
import re
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus.reader import CategorizedPlaintextCorpusReader as Reader


class PolarityDataReader(object):
    """
    PolarityDataReader:
        Reader for POS/NEG Categorized Sentiword data

    uses:
        nltk.corpus.reader.CategorizedPlaintextCorpusReader

    usage:
        
        dataReader = PolarityDataReader([rootLocation],[readerObject])
        dataReader.getDocuments()
        dataReader.setTerms([No:ofTerms])

        featuresets = dataReader.getTermDocMatrix()

    """
    
    def __init__(self, rootLocation = config.POLARITY_DATASET,reader=None):
        super(PolarityDataReader, self).__init__()
        if reader == None:
            self.reader = Reader(rootLocation,r'.*/.*', cat_pattern=r'(.*)/.*')
        else:
            self.reader = reader
        self.setStopWords()
        self.documents = None;
        self.terms = None;


    def getDocuments(self):
        if not self.documents:
            self.documents = [(list(self.reader.words(fileid)), category) 
                for category in self.reader.categories()
                for fileid in self.reader.fileids(category)]
        return self.documents;

    def setStopWords(self,fileLocation = config.STOP_WORDS_FILE):
        stopfile = open(fileLocation, 'r')
        self.stopwords = stopfile.read().split()

    def removeStopWords(self,wordList):
        """ Remove common words which have no search value """
        return [word for word in wordList if word not in self.stopwords]

    def setTerms(self,size=2000,featureSelection='PD',removeStopWords=True):
        if featureSelection == 'PD':
            self.__setTermsPD__(size)
            print "Feature Selection : PD :done "
        
        elif featureSelection == 'CHI_SQUARE':
            self.__setTermsCHISQUARE__(size)
            print "Feature Selection : CHI_SQUARE :done "
        
        else:
            """
            geting most frequent Words
            """
            all_words = [w.lower() for w in self.reader.words()];
            if removeStopWords:
                all_words = self.removeStopWords(all_words);
            all_words = FreqDist(w for w  in all_words)
            self.terms = all_words.keys()[:size]
            print "Feature Selection: frequent Words :done "


    def documentFeatures(self,document,sentiwordnet=False):
        document_words = set(document)
        features = {}
        if sentiwordnet:
            pass
            #TODO
        else :
            for word in self.terms:
                features[word] = (word in document_words)
        return features
       

    def getTermDocMatrix(self):
        return [(self.documentFeatures(document), category) 
            for (document,category) in self.documents]

    def __setTermsPD__(self,size):
        """
        score=|(posDF-negDF)|/(posDF+negDF)
        """
        posWord = {};
        negWord = {};
        
        for word in self.reader.words(categories = ['pos']):
            inc(posWord,word.lower());
        for word in self.reader.words(categories = ['neg']):
            inc(negWord,word.lower());
        
        wordScores = {}
        for word in self.reader.words():
            try:
                posScore = posWord[word]
            except KeyError, e:
                posScore = 0
            try:
                negScore = negWord[word]
            except KeyError, e:
                negScore = 0
            totalScore = posScore + negScore
            if totalScore <= 10 : # min total count
                wordScores[word] = 0.1
            else :
                wordScore[word] = abs(posScore-negScore)/totalScore
        #removeStopWords does no affect accurcy          
        termScore = sorted(wordScores.items(),key=lambda(w,s):s,reverse=True)[:size]
        self.terms = [w for (w,s) in termScore];

    def __setTermsCHISQUARE__(self,size):
        word_fd = FreqDist()
        label_word_fd = ConditionalFreqDist()
            
        for word in self.reader.words(categories=['pos']):
            word_fd.inc(word.lower())
            label_word_fd['pos'].inc(word.lower())

        for word in self.reader.words(categories=['neg']):
            word_fd.inc(word.lower())
            label_word_fd['neg'].inc(word.lower())
            
        pos_word_count = label_word_fd['pos'].N()
        neg_word_count = label_word_fd['neg'].N()
        total_word_count = pos_word_count + neg_word_count

        wordScores = {}
 
        for word, freq in word_fd.iteritems():
            pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
                (freq, pos_word_count), total_word_count)
            neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
                (freq, neg_word_count), total_word_count)
            wordScores[word] = pos_score + neg_score

        termScore = sorted(wordScores.items(),key=lambda(w,s):s,reverse=True)[:size]
        self.terms = [w for (w,s) in termScore];

    def __del__(self):
        pass


class SpecialTokenizer(RegexpTokenizer):
    """
    Tokenizer for Adding the tag NOT_ to every word 
    between a negation word and the first punctuation mark following 
    the negation word

    Super Class : nltk.tokenize.regexp.RegexpTokenizer

    """
    def __init__(self, pattern=r'\w+|[^\w\s]+'):
        super(SpecialTokenizer, self).__init__(pattern)
    def tokenize(self, text):
        tok = super(SpecialTokenizer,self).tokenize(text)
        try:
            ind = tok.index('not')
        except Exception, e:
            return tok
        tok.remove('not')
        for (i,s) in enumerate(tok[ind:]):
            if re.match("^[A-Za-z0-9]*$",s):
                tok[ind + i] = 'NOT_'+s
        return tok


class SpecialPolarityDataReader(PolarityDataReader):
    
    """
    SpecialPolarityDataReader uses Special Tokenizer
    for adding the tag NOT_ to every word between a
    negation word and the first punctuation mark following 
    the negation word

    Super Class : PolarityDataReader

    """

    def __init__(self, rootLocation = config.POLARITY_DATASET):
        reader = Reader(rootLocation,r'.*/.*', cat_pattern=r'(.*)/.*',
            word_tokenizer = SpecialTokenizer())
        super(SpecialPolarityDataReader, self).__init__(reader=reader)


def inc(wordDict,word):
    try:
        wordDict[word] +=1; 
    except KeyError, e:
        wordDict[word] =1;