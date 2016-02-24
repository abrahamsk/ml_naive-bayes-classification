#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# HW 4: Naive Bayes Classification
# Katie Abrahams, abrahake@pdx.edu
# 2/25/16

from __future__ import division
from input import *

"""
2.
A) Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in the training data.
As described in part 1, P(1) should be about 0.4.

B) For each of the 57 features, compute the mean and standard deviation in the training set
of the values given each class.
-See input.py for mean and standard dev calculations

Training data information includes mean and std dev for each of the 57 attributes
and 2 possible class values, for a total of 114 (57*2)
"""

#######################################################################

# A)
# Prior prob. P(1) for positive class and P(0) for negative class
# in training data:
count_spam = 0
count_not_spam = 0
# iterate through matrix to count spam/not spam instances
for row in X_training:
    if row[57] == 1:
        count_spam += 1
    else:
        count_not_spam += 1
# divide num spam instances by number of rows in training
# to calculate class probabilities for training data
prior_prob_spam = count_spam/len(X_training)  # 0.3939
prior_prob_not_spam = count_not_spam/len(X_training)  # 0.6060





