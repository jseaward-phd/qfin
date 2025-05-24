#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:08:29 2025

@author: rakshat

Metrics from http://arxiv.org/abs/2202.00599 not found in Keras
"""


def sesd(y_true, y_pred):
    from tensorflow import math

    sq = (y_true - y_pred) ** 2
    return math.reduce_std(sq)


def mr(y_true, y_pred):
    from tensorflow import math, cast, float32

    return math.reduce_mean(cast(y_pred, float32)) / math.reduce_mean(
        cast(y_true, float32)
    )


def sdr(y_true, y_pred):
    from tensorflow import math

    return math.reduce_std(y_pred / y_true)


def qmse(y_true, y_pred):
    from pennylane.math import stack
    from pennylane.numpy import mean

    return mean((y_true - stack(y_pred)) ** 2)


if __name__ == "__main__":
    pass
