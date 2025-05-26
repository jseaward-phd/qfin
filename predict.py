#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 17:25:25 2025

@author: rakshat
"""

# Should take in a CONFIG ppath as the argument, do all the collection and working out from that and then print the results.


def get_fresh_data(*tickers, scaler=None):  # needs to use tthe different scalers...
    raw_data = data.get_yfin_data(*tickers, start=start)


def main(args):
    CONFIG = yaml.safe_load(open(args.config_path, "r"))


if __name__ == "__main__":
    import argparse
    import data

    parser = argparse.ArgumentParser(
        prog="Market predictions using BiLSTM and/or PQC",
        description="Predicts the next trading day's performance basee don the last 16.",
    )
    parser.add_argument(
        "config_path",
        help="Path to config yaml file used for training.",
        default=".results/passed_config.yaml",
    )

    args = parser.parse_args()
    main(args)
