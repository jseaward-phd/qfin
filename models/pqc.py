#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 09:38:16 2025

@author: rakshat
"""

import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import utils
from tqdm import trange


# horizon set at 1, layers at 6, as in paper
# need to parrallelize/batch this correctly
def build_pqn_and_params(input_len: int = 16):

    dev = qml.device("default.qubit", wires=input_len + 1)
    read_out_wire = input_len
    
    def weighted_rot_bock(params):
        x,z,y = params
        for wire, angle in enumerate(x):
            qml.ops.op_math.Controlled(qml.RX(angle, read_out_wire), wire)
        for wire, angle in enumerate(z):
            qml.ops.op_math.Controlled(qml.RZ(angle, read_out_wire), wire)
        for wire, angle in enumerate(y):
            qml.ops.op_math.Controlled(qml.RY(angle, read_out_wire), wire)

    @qml.qnode(dev)
    def circuit(inputs, params):  # assumes data is scaled between 0 and 1

        # for wire, day in enumerate(inputs):
        #     qml.X(wire) ** day
        qml.AngleEmbedding(inputs.squeeze(),list(range(input_len)))

        qml.layer(weighted_rot_bock,2,params)
        # x0, z0, y0, x1, z1, y1 = params
        # for wire, angle in enumerate(x0):
        #     qml.ops.op_math.Controlled(qml.RX(angle, read_out_wire), wire)
        # for wire, angle in enumerate(z0):
        #     qml.ops.op_math.Controlled(qml.RZ(angle, read_out_wire), wire)
        # for wire, angle in enumerate(y0):
        #     qml.ops.op_math.Controlled(qml.RY(angle, read_out_wire), wire)
        # for wire, angle in enumerate(x1):
        #     qml.ops.op_math.Controlled(qml.RX(angle, read_out_wire), wire)
        # for wire, angle in enumerate(z1):
        #     qml.ops.op_math.Controlled(qml.RZ(angle, read_out_wire), wire)
        # for wire, angle in enumerate(y1):
        #     qml.ops.op_math.Controlled(qml.RY(angle, read_out_wire), wire)

        return qml.expval(qml.Z(read_out_wire))

    params = qml.math.stack([np.random.rand(3, input_len),
                             np.random.rand(3, input_len)],requires_grad=True)* np.pi 
    return circuit, params


def mse(y_true, y_pred):
    return np.mean((y_true - qml.math.stack(y_pred)) ** 2)


class PQN:
    def __init__(
        self,
        input_len: int = 16,
        loss_fn=mse,
        optimizer=qml.optimize.AdamOptimizer,
        metrics=[],
    ):
        self.pqn, self.params = build_pqn_and_params(input_len=input_len)
        self.opt = optimizer()
        self.loss_fn = loss_fn

    def cost(self, batch_x, params, batch_y):
        pred_y = []
        for x_sample in batch_x:
            pred_y.append(self.pqn(x_sample.squeeze(), params))
        pred_y = np.array(pred_y, requires_grad=True)
        return self.loss_fn(batch_y, pred_y)

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

        if validation_split and validation_data is None:
            # Create the validation data using the training data.
            val_size = max(1, int(len(x) * validation_split))
            x, val_x, y, val_y = train_test_split(x, y, test_size=val_size)

        elif validation_data is not None:
            val_x, val_y = validation_data
        else:
            val_x = None

        val_loss = []
        self.best_weights = None

        for epoch in trange(initial_epoch, epochs, unit="Epoch"):
            for batch_num in trange(0, len(x), batch_size, leave=False, unit="batch"):
                batch_train_x = x[batch_num : batch_num + batch_size]
                batch_train_y = y[batch_num : batch_num + batch_size]

                (_, self.params, _), cost = self.opt.step_and_cost(
                    self.cost, batch_train_x, self.params, batch_train_y
                )
                print(f"Train batch {batch_num//batch_size+1} cost: {cost:.3f}\n")

            if shuffle:
                x, y = utils.shuffle(x, y)
            # Run validation.
            if val_x is not None and epoch % validation_freq == 0:
                val_cost = self.cost(batch_train_x, self.params, batch_train_y)
                if epoch > 0:
                    if val_cost < min(val_loss):
                        self.best_weights = self.params
                val_loss.append(val_cost)
                print(f"\nEpoch {epoch} val loss: {val_cost:.3f}")

        return self.params if val_x is None else self.best_weights

    def predict(self, x):
        if x.ndim > 2:
            pred_y = []
            for x_sample in x:
                pred_y.append(self.pqn(x_sample.squeeze(), self.params))
            return np.array(pred_y, requires_grad=True)
        else:
            return self.pqn(x.squeeze(), self.params)

    def evaluate(self, x, y):
        pred_y = self.predict(x)
        return self.loss_fn(y, pred_y)
