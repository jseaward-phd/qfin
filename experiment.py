#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:40:34 2025

@author: rakshat
"""

import os
import yaml

import pandas as pd
import argparse

import data
from models.BiLSTM import build_BLSTM
from models.pqc import PQN
from models import metrics

# parser = argparse.ArgumentParser(
#     prog="BiLSTM vs. PQC training/validation",
#     description="Builds, trains, tests, and compares a bidirectional long short-term memory model to a paramertized quantum circuit for forcasting changes in stock prices. Based on the 2022 parper from Barclays, arXiv:2202.00599.",
# )
# parser.add_argument(
#     "config_path", help="Path to config yaml file.", default="./CONFIG.yaml"
# )


# args = parser.parse_args()

METRICS = {
    "mse": "mse",
    "qmse": metrics.qmse,
    "sesd": metrics.sesd,
    "mr": metrics.mr,
    "sdr": metrics.sdr,
}
CONFIG = yaml.safe_load(open("CONFIG.yaml", "r"))
# CONFIG = yaml.safe_load(open(args.config_path, "r"))

# data setup
# check if data path is a dir or file
data_params = CONFIG["data"]
tickers = data_params["ticker_symols"]
start = data_params["start"]
end = data_params["end"]
seq_length = data_params["past_len"]
f_range = tuple(data_params["scale_range"])

if data_params["data_path"] is not None:
    raise NotImplementedError("Use the yahoo finance api tools for now")
    # raw_data = data.parse_json_data(CONFIG['data']['data_path'])
else:
    raw_data = data.get_yfin_data(*tickers, start=start, end=end)
    data_dict_BLSTM = data.datasets_from_multiindex(
        raw_data, seq_length=seq_length, f_range=f_range, **data_params["kwargs"]
    )
    data_dict_PQC = data.datasets_from_multiindex(
        raw_data,
        seq_length=seq_length,
        f_range=f_range,
        quantum=True,
        **data_params["kwargs"]
    )

# BiLSTM
# setup
init_params = CONFIG["models"]["BiLSTM"]["init"]
pt_path = init_params["pretrained_path"]

metric_fns = [METRICS[x] for x in init_params["metrics"] + init_params["add_metrics"]]

model = build_BLSTM(
    input_len=init_params["past_len"],
    summary=init_params["summary"],
    metrics=metric_fns,
)
if pt_path:
    model.load_weights(pt_path)

# train
# TODO: add checkpointing callback fn
train_params = CONFIG["models"]["BiLSTM"]["train"]
if train_params["train"]:
    model.fit(x=data_dict_BLSTM)

# PQC
init_params = CONFIG["models"]["PQC"]["init"]
metric_fns = [METRICS[x] for x in init_params["metrics"] + init_params["add_metrics"]]
