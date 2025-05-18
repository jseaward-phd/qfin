#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 13:00:30 2025

@author: rakshat
"""

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers
from .metrics import sesd, mr, sdr


# build model
# TODO: config setting/file
def build_BLSTM(input_len: int = 94, horizon: int = 1, summary: bool = True):
    inputs = K.Input(shape=[input_len, 1])
    x = layers.Bidirectional(
        layers.LSTM(128, activation="tanh", return_sequences=True), merge_mode="sum"
    )(inputs)
    x32 = layers.Bidirectional(
        layers.LSTM(32, activation="tanh", return_sequences=True), merge_mode="sum"
    )(x)
    x1 = layers.Bidirectional(
        layers.LSTM(horizon, activation="linear"), merge_mode="sum"
    )(x)
    x32 = layers.Bidirectional(
        layers.LSTM(horizon, activation="linear"), merge_mode="sum"
    )(x32)
    output = layers.Add()([x1, x32])

    model = K.Model(inputs, output)
    if summary:
        model.summary()
    model.compile(
        loss=tf.losses.mse,
        optimizer=K.optimizers.Adam(),
        metrics=["mse", sesd, mr, sdr],
    )
    return model


if __name__ == "__main__":
    pass
