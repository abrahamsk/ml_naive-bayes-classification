#!/usr/bin/env python
# coding=utf-8

# Machine Learning 445
# HW 4: Naive Bayes Classification
# Katie Abrahams, abrahake@pdx.edu
# 2/25/16

from probabilistic_model import *

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