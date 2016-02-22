#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# HW 4: Naive Bayes Classification
# Katie Abrahams, abrahake@pdx.edu
# 2/25/16

import pandas as pd
from sklearn import preprocessing
import numpy as np

# Use all the data (4,601 instances). The full data set has approximately 40% spam, 60% not-spam #

# read in spambase data to a pandas dataframe
df = pd.read_csv('/Users/katieabrahams/PycharmProjects/machinelearningHW4/src/spambase/spambase.data', header=None)
"""
1. Create training and test set:
Split the data into a training and test set. Each of these should have about 2,300 instances,
and each should have about 40% spam, 60% not-spam, to reflect the statistics of the full data set.
40% spam = 920 spam instances
60% not spam = 1380 non-spam instances
"""

# positive: spam
# negative: not spam
df_pos = (df.loc[df[57] == 1])
df_neg = (df.loc[df[57] == 0])
# print df_pos.shape  # (1813, 58) 1/2(1812) = 906
# print df_neg.shape # (2788, 58) 1/2(2788) = 1394

###########################################################################

# concat pos + neg dataframe subsets for training and test data

# training data
frames_training = [df_pos[0:906], df_neg[0:1394]]
df_training = pd.concat(frames_training)
df_training = df_training.reset_index(drop=True)

# test data
frames_test = [df_pos[906:1812], df_neg[1394:2788]]
df_test = pd.concat(frames_test)
df_test = df_test.reset_index(drop=True)
# print df_training.shape  # (2300, 58)
# print df_test.shape  # (2300, 58)

###########################################################################

# shuffle training data #
# frac=1 means return all rows in random order
df_training = df_training.sample(frac=1).reset_index(drop=True)

###########################################################################
#
# # convert dataframe into a numpy matrix
# X = df_training.as_matrix().astype(np.float)
# # scale data #
# # preprocess everything in training data matrix except the last column (1 or 0 to identify spam or not)
# # then concat the identifying column to the preprocessed training data
# scaler = preprocessing.StandardScaler()
# X_to_scale = X[:,:57].copy()
# X_scaled = scaler.fit_transform(X_to_scale)
# X_col = X[:,57]
# X_concat = np.concatenate((X_scaled, X_col[None].T), axis=1)
# np.savetxt("/Users/katieabrahams/PycharmProjects/machinelearningHW4/src/sklearn_svm/numpy_train.csv", X_concat,
#            delimiter=",")
#
# # Scale test data using standardization parameters from training data
# X_test = df_test.as_matrix().astype(np.float)
# X_test_to_scale = X_test[:,:57].copy()
# X_test_scaled = scaler.fit_transform(X_test_to_scale)
# X_test_col = X_test[:,57]
# X_test_concat = np.concatenate((X_test_scaled, X_test_col[None].T), axis=1)
# np.savetxt("/Users/katieabrahams/PycharmProjects/machinelearningHW4/src/sklearn_svm/numpy_test.csv", X_test_concat,
#            delimiter=",")
