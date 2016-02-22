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
# print df_training  # (2300, 58)
# print df_test.shape  # (2300, 58)

# convert dataframes into a numpy matrix
X_training = df_training.as_matrix().astype(np.float)
X_test = df_test.as_matrix().astype(np.float)

###########################################################################

# shuffle training data #
# frac=1 means return all rows in random order
# df_training = df_training.sample(frac=1).reset_index(drop=True)

###########################################################################
