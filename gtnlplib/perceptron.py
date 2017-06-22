from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector,argmax

def perceptron_update(x,y,weights,labels):
    """compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    update = defaultdict(float)
    max_label, scores = predict(x, weights, labels)
    if max_label != y:
        y_features = make_feature_vector(x, y)
        max_label_features = make_feature_vector(x, max_label)
        
        for key in y_features:
            update[key] = y_features[key] - (max_label_features[key] if key in max_label_features else 0.0)

        for key in max_label_features:
            update[key] = (y_features[key] if key in y_features else 0.0) - max_label_features[key]

    return update

def estimate_perceptron(x,y,N_its):
    """estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in xrange(N_its):
        for x_i,y_i in zip(x,y):
            update = perceptron_update(x_i, y_i, weights, labels)
            for key in update:
                weights[key] += update[key]
        weight_history.append(weights.copy())
    return weights, weight_history

def estimate_avg_perceptron(x,y,N_its):
    """estimate averaged perceptron classifier

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    w_sum = defaultdict(float)
    weights = defaultdict(float)
    avg_weights = defaultdict(float)
    weight_history = []
    
    t = 1.0
    for it in xrange(N_its):
        for x_i,y_i in zip(x,y):
            update = perceptron_update(x_i, y_i, weights, labels)
            for key in update:
                w_sum[key] += t * update[key]
                weights[key] += update[key]
            t += 1
        for key in weights:
            avg_weights[key] = weights[key] - 1 / t * w_sum[key]
        weight_history.append(avg_weights.copy())
    return avg_weights, weight_history