#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# HW 4: Naive Bayes Classification
# Katie Abrahams, abrahake@pdx.edu
# 2/25/16

from probabilistic_model import *
import math
import timing


"""
3.
Run Naïve Bayes on the test data.
- Use the Naïve Bayes algorithm to classify the instances in your test set,
using P(xi |cj)=N(xi;μi,cj,σi,cj )
where N(x;μ,σ)= [1/(sqrt(2π)σ)]*e^[−((x−μ)^2)/(2σ^2)]

Because a product of 58 probabilities will be very small, we will instead use the log of the product.
Recall that the classification method is:
classNB(x)=argmax[P(class)∏P(xi |class)]

Since
argmax f(z) = argmax log f(z)
we have:
classNB(x)=argmax[(class)∏P(xi |class)]
"""

###########
# functions
###########


def gaussian_probability(x, mean, std_dev):
    """
    Use NB to classify instances in test set
    using Gaussian normal distribution function N
    math.exp(x) returns e**x.
    :param x:
    :param mean:
    :param std_dev:
    :return:
    """
    if(std_dev == 0):
        std_dev = .01
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std_dev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std_dev)) * exp


def predict_class():
    """
    Predict one row in test data
    :return:
    """


def predict_all():
    """
    Predict classes for test data
    Use X_test_features, prior_prob_spam, prior_prob_not_spam
    From input and probabilistic_model files
    Calculate argmax for spam and not spam classes
    :return list of class predictions:
    """
    # store the test data predictions
    classes = []
    for i in range(len(X_test_features)):
        prediction = []
        sum_prob = [gaussian_probability(X_test_features[i][j], means[i], std_devs[i]) for j in X_test_features[i][j]]

        predict_spam = np.log10(prior_prob_spam) + np.log10(
            gaussian_probability(X_test_features[i], means[i], std_devs[i]))

        predict_not_spam = np.log10(prior_prob_not_spam) + np.log10(
            gaussian_probability(X_test_features[i], means[i], std_devs[i]))

        prediction.append(predict_spam, predict_not_spam)
        class_nb = max(prediction)
        classes.append(class_nb)
    return classes


# def getPredictions(summaries, testSet):
# 	predictions = []
# 	for i in range(len(testSet)):
# 		result = predict(summaries, testSet[i])
# 		predictions.append(result)
# 	return predictions