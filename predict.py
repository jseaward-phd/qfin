#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 17:25:25 2025

@author: rakshat
"""

# Should take in a CONFIG ppath as the argument, do all the collection and working out from that and then print the results.
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import data
import pickle
import datetime


# disable (some) keras warnings:
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from models.pqc import PQN
from models.BiLSTM import build_BLSTM


def get_fresh_data(tickers, start, colname):  # needs to use tthe different scalers...
    raw_data = data.get_yfin_data(*tickers, start=start)
    out_dict = dict.fromkeys(raw_data.columns.levels[1])
    for ticker in tickers:
        out_dict[ticker] = raw_data["Close"][ticker].dropna()[-past_len:]
    return out_dict


def parse_data_file(ticker, model):
    data_path = os.path.join(root_dir, ticker, model, "data_dict.pkl")
    data_dict = pickle.load(open(data_path, "rb"))
    return data_dict["test_x"], data_dict["test_y"], data_dict["scaler"]


def load_blstm(ticker, weight_fn="trained.weights.h5"):
    model = build_BLSTM(input_len=past_len, summary=False)
    pt_weight_path = os.path.join(root_dir, ticker, "BiLSTM", weight_fn)
    model.load_weights(pt_weight_path)
    return model


def main(args):
    CONFIG = yaml.safe_load(open(args.config_path, "r"))

    global root_dir, past_len
    root_dir = os.path.split(args.config_path)[0]
    past_len = CONFIG["data"]["lengths"]["past_len"]

    if args.days < past_len:
        args.days = int(CONFIG["data"]["lengths"]["past_len"] * 1.5)
        print(
            f"INFO: Passed too few days to grab. Changed to 1.5x context length, {args.days:d}"
        )

    delta_t = datetime.timedelta(days=args.days)
    now = datetime.datetime.now()
    start_dt = now - delta_t
    start = start_dt.strftime("%Y-%m-%d")

    colname = CONFIG["data"]["kwargs"]["colname"]
    tickers = CONFIG["data"]["ticker_symols"]
    fresh_data_dict = get_fresh_data(tickers, start, colname)

    config_model_names = [x for x in CONFIG["models"].keys() if x != "common"]
    blstm_weight_fn = CONFIG["models"]["BiLSTM"]["train"]["weight_save_filename"]
    pqc_weight_fn = CONFIG["models"]["PQC"]["train"]["weight_save_filename"]
    pqc_blocks = CONFIG["models"]["PQC"]["init"]["blocks"]

    blstm_model = build_BLSTM(input_len=past_len, summary=False)

    print(f"\nPredicting for ticker symbols: {', '.join(tickers)}...")
    for ticker in tickers:
        print(f"\nTicker symbol {ticker}:")
        models = os.listdir(os.path.join(root_dir, ticker))

        for model_name in models:
            test_x, test_y, scaler = parse_data_file(ticker, model_name)

            if model_name == "BiLSTM":
                pt_weight_path = os.path.join(
                    root_dir, ticker, model_name, blstm_weight_fn
                )
                blstm_model.load_weights(pt_weight_path)
                print("BiLSTM Unscaled evaluation:")
                blstm_model.evaluate(test_x, test_y)
                dat = fresh_data_dict[ticker].to_numpy()
                if scaler is not None:
                    dat = scaler.transform(dat.reshape([-1, 1]))
                prediction = blstm_model(dat.reshape([1, -1])).numpy()
                if scaler is not None:
                    prediction = scaler.inverse_transform(prediction.reshape([1, -1]))
                print(
                    f"BiLSTM prediction for {colname} of {ticker} on next trading day: {prediction.item():.5f}"
                )
            elif model_name == "PQC":
                pqc_model = PQN(input_len=past_len, blocks=pqc_blocks, scaler=scaler)
                pqc_model.load(
                    os.path.join(root_dir, ticker, model_name, pqc_weight_fn)
                )
                print("PQC Unscaled evaluation:")
                _ = pqc_model.evaluate(test_x, test_y, rescale=False, verbose=True)
                dat = fresh_data_dict[ticker].to_numpy()
                if scaler is not None:
                    dat = scaler.transform(dat.reshape([-1, 1]))
                prediction = pqc_model.predict(
                    dat,
                    rescale=(scaler is not None),
                ).item()

                print(
                    f"PQC prediction for {colname} of {ticker} on next trading day: {prediction:.5f}"
                )
            else:
                print(
                    f"Uknown model type {model_name}. Expected one of {config_model_names}."
                )


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        prog="Market predictions using BiLSTM and/or PQC",
        description="Predicts the next trading day's performance basee don the last 16.",
    )
    parser.add_argument(
        "config_path",
        help="Path to config yaml file used for training.",
        default=".results/passed_config.yaml",
    )
    parser.add_argument(
        "--days",
        "-d",
        help="Number of days back to grab data, for safety. Number feed to the model is always taken from the passed config file.",
        type=int,
        default=30,
    )

    args = parser.parse_args()
    main(args)
