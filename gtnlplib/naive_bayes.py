from __future__ import division
from gtnlplib.preproc import get_corpus_counts
from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation
import numpy as np
import math
from collections import defaultdict

def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_counts = defaultdict(float)
    for pos, curr_label in enumerate(y):
        if curr_label == label:
            for word in x[pos]:
                corpus_counts[word] += x[pos][word]
    return corpus_counts
    
def estimate_pxy(x,y,label,smoothing,vocab):
    """Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    log_probabilities = defaultdict(float)
    corpus_counts = get_corpus_counts(x, y, label)
    total = sum(corpus_counts.values())
    for word in vocab:
        log_probabilities[word] = math.log(((corpus_counts[word] if word in corpus_counts else 0) + smoothing) / (total + len(vocab) * smoothing))
    return log_probabilities
    
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict

    """
    labels = set(y)
    doc_counts = defaultdict(float)
    weights = defaultdict(float)

    vocab = set()
    for base_features in x:
        for word in base_features.keys():
            vocab.add(word)

    for label in y:
        doc_counts[label] += 1


    for label in labels:
        weights[(label, OFFSET)] = math.log(doc_counts[label] / sum(doc_counts.values()))
        log_probabilities = estimate_pxy(x, y, label, smoothing, vocab)
        for word in log_probabilities:
            weights[(label, word)] = log_probabilities[word]

    return weights
    
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values to try
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    """
    scores = {}
    labels = set(y_dv)
    for smoothing in smoothers:
        weights = estimate_nb(x_tr, y_tr, smoothing)
        y_hat = clf_base.predict_all(x_dv, weights, labels)
        scores[smoothing] = evaluation.acc(y_hat,y_dv)
    return clf_base.argmax(scores), scores