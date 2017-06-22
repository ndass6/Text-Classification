from gtnlplib.constants import OFFSET
from collections import Counter
import numpy as np

argmax = lambda x : max(x.iteritems(),key=lambda y : y[1])[0]

def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    feature_vector = { (label, OFFSET) : 1}
    for feature in base_features:
        feature_vector[(label, feature)] = base_features[feature]
    return feature_vector
    
def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    scores = {}
    for label in labels:
        scores[label] = 0.0
        for feature in base_features:
            scores[label] += base_features[feature] * weights[(label, feature)]
        scores[label] += weights[(label, OFFSET)]
    return argmax(scores), scores

def predict_all(x,weights,labels):
    """Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    """
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat

def get_top_features_for_label(weights,label,k=5):
    """Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    """
    filtered_weights = {key:value for key, value in weights.iteritems() if label in key[0] and value}
    return Counter(filtered_weights).most_common(k)