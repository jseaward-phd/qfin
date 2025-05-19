#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 09:38:16 2025

@author: rakshat
"""

import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


# TODO: extend horizon
def build_pqn_and_params(input_len: int = 16, horizon: int = 1, layers: int = 6):
    assert layers % 3 == 0, "model architectures uses triplets of layers"
    dev = qml.device("default.qubit", wires=input_len + horizon)
    read_out_wire = input_len + horizon - 1

    @qml.qnode(dev)
    def circuit(inputs, params):  # assumes data is scaled between 0 and 1

        for wire, day in enumerate(inputs):
            qml.X(wire) ** day

        for i in range(0, params.shape[0], 3):
            x, z, y = params[i : i + 3, :]
            for wire, angle in enumerate(x):
                qml.ops.op_math.Controlled(qml.RX(angle, read_out_wire), wire)
            for wire, angle in enumerate(z):
                qml.ops.op_math.Controlled(qml.RZ(angle, read_out_wire), wire)
            for wire, angle in enumerate(y):
                qml.ops.op_math.Controlled(qml.RY(angle, read_out_wire), wire)

        return qml.expval(qml.Z(read_out_wire))

    params = np.random.rand(layers, input_len, requires_grad=True) * np.pi * 2
    return circuit, params


def data_gen(x, y):
    for a, b in zip(x, y):
        yield a, b


class PQN:
    def __init__(
        self,
        input_len: int = 16,
        layers: int = 6,
        loss_fn=tf.losses.mse,
        optimizer=qml.optimize.AdamOptimizer(),
        metrics=[],
    ):
        self.pqn, self.params = build_pqn_and_params(input_len=input_len, layers=layers)
        self.opt = optimizer
        self.loss_fn = loss_fn

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        # verbose="auto",
        # callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        # class_weight=None,
        # sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):

        if batch_size is None:
            batch_size = len(y)

        train_ds = tf.data.Dataset.from_generator(
            data_gen, args=(x, y), output_types=tf.float32
        )

        if validation_split and validation_data is None:
            # Create the validation data using the training data.
            val_size = max(1, int(len(x) * validation_split))
            x, val_x, y, val_y = train_test_split(x, y, test_size=val_size)

        for epoch in range(initial_epoch, epochs):

            # Override with model metrics instead of last step logs if needed.
            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Run validation.
            if validation_data is not None and self._should_eval(
                epoch, validation_freq
            ):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = TFEpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        distribute_strategy=self.distribute_strategy,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {"val_" + name: val for name, val in val_logs.items()}
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if isinstance(self.optimizer, optimizers_module.Optimizer) and epochs > 0:
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history
