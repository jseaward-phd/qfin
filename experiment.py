#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:40:34 2025

@author: rakshat
"""

import os
import yaml
import pickle

# import pandas as pd # TODO: save results as csv

from models import metrics

from contextlib import redirect_stdout


METRICS = {
    "mse": "mse",
    "qmse": metrics.qmse,
    "sesd": metrics.sesd,
    "mr": metrics.mr,
    "sdr": metrics.sdr,
}

def blstm_loop(blstm_config, data_dict):
    from models.BiLSTM import build_BLSTM

    # BiLSTM
    # setup
    init_params = blstm_config["init"]
    init_params["metrics"] = init_params["metrics"] + init_params["add_metrics"]
    del init_params["add_metrics"]
    train_params = blstm_config["train"]

    metric_fns = [METRICS[x] for x in init_params["metrics"]]

    for ticker in data_dict.keys():
        print(f"Processing {ticker} data...")
        stock_path = os.path.join(init_params["save_dir"], ticker, "BiLSTM")
        if not os.path.exists(stock_path):
            os.makedirs(stock_path)

        data_in = data_dict[ticker]
        pickle.dump(data_in, open(os.path.join(stock_path, "data_dict.pkl"), "wb"))
        with open(os.path.join(stock_path, "log.txt"), "a") as f:
            with redirect_stdout(f):
                model = build_BLSTM(
                    input_len=init_params["past_len"],
                    summary=init_params["summary"],
                    metrics=metric_fns,
                )

                pt_weight_path = os.path.join(stock_path, init_params["pt_filename"])
                if train_params["load_pretrained"]:
                    print(f"Loading weights from {pt_weight_path}...")
                    try:
                        model.load_weights(pt_weight_path)
                    except Exception as e:
                        print(
                            f"Exception: {e} encoutered while trying to load weights. Are the weights there?"
                        )

                # train
                # TODO: Expects validatin data
                """
                TODO:
                callbacks for logging (e.g. tf.keras.callbacks.TensorBoard(log_dir='./logs') ),
                lr sceduler (e.g. tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch / 20))),
                and early stopping (e.g. tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True) )
                are passed to the callbacks argument of model.fit as a list
                """

                if train_params["train"]:
                    weight_save_path = os.path.join(
                        stock_path, train_params["weight_save_filename"]
                    )
                    train_args_dict = {
                        "x": data_in["train_x"],
                        "y": data_in["train_y"],
                        "epochs": train_params["epochs"],
                        "validation_data": (data_in["val_x"], data_in["val_y"]),
                    }
                    print(f"Training paramters: {train_args_dict}\n")
                    model.fit(**train_args_dict)
                    model.save_weights(weight_save_path)
                    print(f"Model weights saved at {weight_save_path}.")

                # test
                if train_params["test"]:
                    print("\nUnscaled evaluation:")
                    model.evaluate(data_in["test_x"], data_in["test_y"])

                    print("\nRescaled evaluation:")
                    y_pred = data_in["scaler"].inverse_transform(
                        model(data_in["test_x"])
                    )
                    y_true = data_in["scaler"].inverse_transform(
                        data_in["test_y"].reshape([-1, 1])
                    )
                    print("True:", y_true)
                    print("Pred:", y_pred)
                    for m in init_params["metrics"]:
                        print(m, ": ", METRICS[m](y_true, y_pred).numpy().item())
    return


def pqc_loop(pqc_config, data_dict):
    from models.pqc import PQN

    # setup
    init_params = pqc_config["init"]
    init_params["metrics"] = init_params["metrics"] + init_params["add_metrics"]
    del init_params["add_metrics"]
    train_params = pqc_config["train"]

    metric_fns = [METRICS[x] for x in init_params["metrics"]]

    for ticker in data_dict.keys():

        print(f"Processing {ticker} data...")
        stock_path = os.path.join(init_params["save_dir"], ticker, "PQC")
        if not os.path.exists(stock_path):
            os.makedirs(stock_path)

        data_in = data_dict[ticker]
        pickle.dump(data_in, open(os.path.join(stock_path, "data_dict.pkl"), "wb"))
        with open(os.path.join(stock_path, "log.txt"), "a") as f:
            with redirect_stdout(f):
                pqc = PQN(
                    input_len=init_params["past_len"],
                    blocks=init_params["blocks"],
                    metrics=metric_fns,
                    scaler=data_in["scaler"],
                    save_path=os.path.join(
                        stock_path, train_params["weight_save_filename"]
                    ),
                )

                pt_weight_path = os.path.join(stock_path, init_params["pt_filename"])
                if train_params["load_pretrained"]:
                    print(f"Loading weights from {pt_weight_path}...")
                    try:
                        pqc.load(pt_weight_path)
                    except Exception as e:
                        print(
                            f"Exception: {e} encoutered while trying to load weights. Are the weights there?"
                        )

                # train
                # TODO: Expects validatin data
                if train_params["train"]:
                    train_args_dict = {
                        "x": data_in["train_x"],
                        "y": data_in["train_y"],
                        "epochs": train_params["epochs"],
                        "validation_data": (data_in["val_x"], data_in["val_y"]),
                        **train_params["kwargs"],
                    }
                    print(f"Training paramters: {train_args_dict}\n")
                    pqc.fit(**train_args_dict)

                # test
                if train_params["test"]:
                    print("\nRescaled output comparison:")
                    [
                        print(a, b.item())
                        for a, b in pqc.compare(
                            data_in["test_x"], data_in["test_y"], rescale=True
                        )
                    ]
                    print("\nRescaled evaluation:")
                    pqc.evaluate(
                        data_in["test_x"], data_in["test_y"], rescale=True, verbose=True
                    )

    return


def main(args):
    CONFIG = yaml.safe_load(open(args.config_path, "r"))

    results_dir = CONFIG["models"]["common"]["init"]["save_dir"]
    if CONFIG["fresh_start"] and os.path.exists(results_dir):
        if len(os.listdir(results_dir)) > 0:
            import shutil

            shutil.rmtree(results_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir, "passed_config.yaml"), "w") as f:
        yaml.dump(CONFIG, f)

    # data setup
    # check if data path is a dir or file
    data_params = CONFIG["data"]
    tickers = data_params["ticker_symols"]
    start = data_params["start"]
    end = data_params["end"]
    seq_length = data_params["lengths"]["past_len"]
    f_range = tuple(data_params["scale_range"])

    if data_params["data_path"] is not None:
        raise NotImplementedError("Use the yahoo finance api tools for now")
        # raw_data = data.parse_json_data(CONFIG['data']['data_path'])
    else:
        raw_data = data.get_yfin_data(*tickers, start=start, end=end)
        # TODO: Error handle failed download
        data_dict_BLSTM = data.datasets_from_multiindex(
            raw_data, seq_length=seq_length, f_range=f_range, **data_params["kwargs"]
        )
        data_dict_PQC = data.datasets_from_multiindex(
            raw_data,
            seq_length=seq_length,
            f_range=f_range,
            quantum=True,
            **data_params["kwargs"],
        )

    blstm_config = CONFIG["models"]["BiLSTM"]
    pqc_config = CONFIG["models"]["PQC"]

    blstm_loop(blstm_config, data_dict_BLSTM)
    pqc_loop(pqc_config, data_dict_PQC)

    print("Done!")

    return


if __name__ == "__main__":
    import argparse
    import data

    parser = argparse.ArgumentParser(
        prog="BiLSTM vs. PQC training/validation",
        description="Builds, trains, tests, and compares a bidirectional long short-term memory model to a paramertized quantum circuit for forcasting changes in stock prices. Based on the 2022 parper from Barclays, arXiv:2202.00599.",
    )
    parser.add_argument(
        "config_path", help="Path to config yaml file.", default="./CONFIG.yaml"
    )

    args = parser.parse_args()
    main(args)
