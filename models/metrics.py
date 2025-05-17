#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:08:29 2025

@author: rakshat

Metrics from http://arxiv.org/abs/2202.00599 not found in Keras
"""

from tensorflow.keras import backend as K


def sesd(y_true, y_pred):
    sq = (y_true - y_pred) ** 2
    return K.std(sq)


def mr(y_true, y_pred):
    return K.mean(y_pred) / K.mean(y_true)


def sdr(y_true, y_pred):
    return K.std(y_pred / y_true)


if __name__ == "__main__":
    pass
