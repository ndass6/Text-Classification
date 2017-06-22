import nltk, re
import pandas as pd
from collections import Counter

def tokenize_and_downcase(string,vocab=None):
    """for a given string, corresponding to a document:
    - tokenize first by sentences and then by word
    - downcase each token
    - return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    bow = Counter()
    for sentence in nltk.sent_tokenize(string):
        for word in nltk.word_tokenize(sentence):
            bow[word.lower()] += 1
    return bow

def read_data(csvfile,labelname,preprocessor=lambda x : x):
    # note that use of utf-8 encoding to read the file
    df = pd.read_csv(csvfile,encoding='utf-8')
    return df[labelname].values,[preprocessor(string) for string in df['text'].values]

def get_corpus_counts(list_of_bags_of_words):
    counts = Counter()
    for bow in list_of_bags_of_words:
        for key,val in bow.iteritems():
            counts[key] += val
    return counts

def custom_preproc(string):
    """for a given string, corresponding to a document, tokenize first by sentences and then by word; downcase each token; return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    bow = Counter()
    regex = re.compile('[^a-zA-Z]')
    for sentence in nltk.sent_tokenize(string):
        for word in nltk.word_tokenize(sentence):
            if 'http' not in word and 'www' not in word:
                word = regex.sub('', word)
                if word:
                    bow[word.lower()] += 1
    return bow
