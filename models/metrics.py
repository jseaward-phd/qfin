#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:08:29 2025

@author: rakshat

Metrics from http://arxiv.org/abs/2202.00599 not found in Keras
"""
from tensorflow import math
from pennylane.math import stack
from pennylane.numpy import mean


def sesd(y_true, y_pred):
    sq = (y_true - y_pred) ** 2
    return math.reduce_std(sq)


def mr(y_true, y_pred):
    return math.reduce_mean(y_pred) / math.reduce_mean(y_true)


def sdr(y_true, y_pred):
    return math.reduce_std(y_pred / y_true)


def qmse(y_true, y_pred):
    return mean((y_true - stack(y_pred)) ** 2)


if __name__ == "__main__":
    pass
