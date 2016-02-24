#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# HW 4: Naive Bayes Classification
# Katie Abrahams, abrahake@pdx.edu
# 2/25/16

from __future__ import division
from probabilistic_model import *
import math
# import timing

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

    # catch div by zero errors
    if (std_dev == 0.0):
        std_dev = .01

    # avoid math domain errors by using log rule:
    # log(a/b) == log(b)-log(a)
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std_dev, 2))))
    denominator = (math.sqrt(2 * math.pi) * std_dev)
    if (exp == 0.0):
        return exp - math.log(denominator)
    else:
        return math.log(exp) - math.log(denominator) * exp


def predict_all():
    """
    Predict classes for test data
    Use X_test_features, prior_prob_spam, prior_prob_not_spam
    From input and probabilistic_model files
    Calculate argmax for spam and not spam classes
    :return list of class predictions:
    """

    # use math.log (base e)
    # use logs and sums rather than products when computing prediction
    predictions = []
    # predict class for each row in test features matrix
    for row in range(len(X_test_features)):
        probabilities_pos = []
        probabilities_neg = []
        # for each item in the instance row (for each feature), calculate gaussian probability using N function
        for i in range(len(X_test_features[row])):
            # log moved to inside gaussian_probability function
            probability_log_pos = gaussian_probability(X_test_features[row,i], pos_means_training[i], pos_std_devs_training[i])
            probabilities_pos.append(probability_log_pos)

            probability_log_neg = gaussian_probability(X_test_features[row,i], neg_means_training[i], neg_std_devs_training[i])
            probabilities_neg.append(probability_log_neg)

        # get prediction for positive and negative classes
        # by summing log of prior probability and sum of gaussian prob for each feature (done above)
        predict_spam = math.log(prior_prob_spam) + sum(probabilities_pos)
        predict_not_spam = math.log(prior_prob_not_spam) + sum(probabilities_neg)

        # assign class prediction based on argmax of positive (spam) and negative (not spam)
        if predict_spam > predict_not_spam:
            predictions.append(1.0)
        else:
            predictions.append(0.0)
    # return list of predictions for spam/not spam for all instances in test set
    return predictions


def get_stats():
    """
    Get stats for test data classifications
    :return:
    """
    # predict classes for spam test data with predict_all()
    predictions = predict_all()

    # print stats
    correct_predictions = 0
    # print [(i,j) for i,j in zip(X_test_classifier, predictions) if i != j]
    for i,j in zip(X_test_classifier, predictions):
        if i==j:
            correct_predictions += 1
    accuracy = correct_predictions / len(X_test_features)
    print "----- Accuracy across the test set -----\n", accuracy, "\nCorrect predictions / 2300 =", \
        correct_predictions, "\n----------------------------------------"

#######################################################################

def main():
    # get stats for test data
    get_stats()


if __name__ == "__main__":
    main()
