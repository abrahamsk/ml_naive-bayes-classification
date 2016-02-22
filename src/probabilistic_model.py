#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# HW 4: Naive Bayes Classification
# Katie Abrahams, abrahake@pdx.edu
# 2/25/16

from __future__ import division
from input import *
import timing

"""
2.
A) Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in the training data.
As described in part 1, P(1) should be about 0.4.

B) For each of the 57 features, compute the mean and standard deviation in the training set
of the values given each class.
"""

############
# functions
############

# returns the last index of the row if it is 1 (i.e. spam)
def pos_spam(row):
    return row[57] == 1

# returns the last index of the row if it is 0 (i.e. not spam)
def neg_not_spam(row):
    return row[57] == 0

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
prior_prob_spam = count_spam/len(X_training)  # 0.3939
prior_prob_not_spam = count_not_spam/len(X_training)  # 0.6060

#######################################################################

# B)
# For each of the 57 features, compute the mean and standard deviation
# in the training set of the values given each class.
means = np.mean(X_training[:,0:57], axis = 0)
std_devs = np.std(X_training[:,0:57], axis = 0)

#######################################################################

# Save to file
np.savetxt("output/training.csv", X_training, delimiter=",")
np.savetxt("output/test.csv", X_test, delimiter=",")
np.savetxt("output/means.csv", means, delimiter=",")
np.savetxt("output/std_devs.csv", std_devs, delimiter=",")



