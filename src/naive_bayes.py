#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# HW 4: Naive Bayes Classification
# Katie Abrahams, abrahake@pdx.edu
# 2/25/16

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
    if std_dev == 0:
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
    ################################################
    # SCRATCH
    #
    # print X_test_features.dtype  # float64
    # for row in X_test_features:
    #     for i in row:
    #         print i
    # print X_test_features
    #
    # for row in X_test_features:
    #     for i in row:
    #         print row, i

    # print range(57) # 0 - 56
    # print range(len(pos_means_training)) # 0 - 56
    # print range(len(X_test_features)) # 0 - 2299

    # count_row = 0
    # count_i = 0
    # for row in X_test_features:
    #     for i in range(len(row)): # 0 - 56
    #         count_i += 1
    #     count_row += 1
    # print count_row # 2300
    # print count_i # 131100

    # for row in X_test_features:
    #     for i in row:
    #         print row, i

    # for row in range(len(X_test_features)):
    #     print len(X_test_features[row])
    ################################################

    # use math.log (base e)
    # use logs and sums rather than products when computing prediction
    classes = []
    # predict class for each row in test features matrix
    for row in range(len(X_test_features)):
        probabilities_pos = []
        probabilities_neg = []
        for i in range(len(X_test_features[row])):
            probabilities_pos.append(math.log(
                gaussian_probability(X_test_features[row,i], pos_means_training[i], pos_std_devs_training[i])))

            probabilities_neg.append(math.log(
                gaussian_probability(X_test_features[row,i], neg_means_training[i], neg_std_devs_training[i])))

        predict_spam = math.log(prior_prob_spam) + sum(probabilities_pos)
        predict_not_spam = math.log(prior_prob_not_spam) + sum(probabilities_neg)

        # assign class based on argmax
        if predict_spam > predict_not_spam:
            classes.append(1.0)
        else:
            classes.append(0.0)

    return classes


# def getPredictions(summaries, testSet):
# 	predictions = []
# 	for i in range(len(testSet)):
# 		result = predict(summaries, testSet[i])
# 		predictions.append(result)
# 	return predictions

#######################################################################

def main():
    # predict classes for spam test data
    predict_all()


if __name__ == "__main__":
    main()
