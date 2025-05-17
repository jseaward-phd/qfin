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
inputs = K.Input(shape=[94, 1])
x = layers.Bidirectional(
    layers.LSTM(128, activation="tanh", return_sequences=True), merge_mode="sum"
)(inputs)
x32 = layers.Bidirectional(
    layers.LSTM(32, activation="tanh", return_sequences=True), merge_mode="sum"
)(x)
x1 = layers.Bidirectional(layers.LSTM(1, activation="linear"), merge_mode="sum")(x)
x32 = layers.Bidirectional(layers.LSTM(1, activation="linear"), merge_mode="sum")(x32)
output = layers.Add()([x1, x32])

model = K.Model(inputs, output)
model.summary()
model.compile(
    loss=tf.losses.mse,
    optimizer=K.optimizers.Adam(),
    metrics=["mse", sesd, mr, sdr],
)

if __name__ == "__main__":
    pass
