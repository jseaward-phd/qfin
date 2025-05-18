#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 09:38:16 2025

@author: rakshat
"""

import pennylane as qml
from pennylane import numpy as np

#TODO: extend horizon
def build_pqn(input_len: int = 16, horizon: int = 1, layers:int = 6):
    assert layers%3 == 0, "model architectures uses triplets of layers"
    dev = qml.device("default.qubit",wires = input_len + horizon)
    read_out_wire =  input_len + horizon - 1
    
    @qml.qnode(dev)
    def circuit(params,data):
        for i in range(0,params.shape[0],3):
            x,z,y = params[i:i+3,:]
            for wire,angle in enumerate(x):
                qml.ops.op_math.Controlled(qml.RX(angle,read_out_wire),wire)
            for wire,angle in enumerate(z):
                qml.ops.op_math.Controlled(qml.RZ(angle,read_out_wire),wire)
            for wire,angle in enumerate(y):
                qml.ops.op_math.Controlled(qml.RY(angle,read_out_wire),wire)
                
        return qml.expval(qml.Z(read_out_wire))
    
    params  = np.random.rand(layers,input_len) * np.pi*2