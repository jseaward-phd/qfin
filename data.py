# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from typing import Union, Tuple
from typing_extensions import Unpack
import os

import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
import pennylane.numpy as qnp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def parse_json_data(path: Union[str, os.PathLike]):
    if os.path.isdir(path):
        out_df = pd.DataFrame()
        for fn in os.listdir(path):
            df = pd.read_json(os.path.join(path, fn))
            out_df = pd.concat([out_df, df])
    else:
        out_df = pd.read_json(path)
    out_df = out_df.sort_values("Date", ascending=True).reset_index(drop=True)

    out_df["Prec_Change"] = out_df.Adj_Close.diff() / out_df.Adj_Close * 100
    return out_df.dropna()


def get_yfin_data(*tickers: str, start=None, end=None) -> pd.DataFrame:
    data = yf.download(
        tickers, auto_adjust=False, start=start, end=end, multi_level_index=True
    )
    for ticker in data.columns.levels[1]:
        data["Perc_Change", ticker] = data["Close", ticker].pct_change()
    return data


def plot_multidf(df: pd.DataFrame) -> None:
    for ticker in df.columns.levels[1]:
        plt.plot(df.index, df["Close"][ticker], ".k")
        plt.title("$" + ticker)
        plt.xlabel("Date")
        plt.ylabel("Split-Adjusted Close Price")
        plt.show()


# we are going to ignore after-hours trading and say it is uniformly sampled in trading-days
def construct_dataset(
    df,
    ticker: Union[str, None] = "AAPL",
    seq_length: int = 94,
    pred_samples: int = 1,
    test_frac: float = 0.25,
    val_frac: float = 0.2,
    colname: str = "Perc_Change",
    preprocess: bool = True,
    f_range: Tuple[float, float] = (0.2, 0.8),
    quantum: bool = False,
) -> Union[
    Tuple[np.ndarray, Unpack[Tuple[np.ndarray, ...]], MinMaxScaler],
    Tuple[np.ndarray, ...],
]:
    if test_frac > 1:
        test_frac = test_frac / len(df)
        print(
            f"test_frac > 1 passed. Assuming it is number of test samples. Set test_frac to {test_frac}."
        )

    if quantum:
        f_range = (0, 1)

    # want to make a random selection of chunks
    chunk_length = seq_length + pred_samples
    num_chunks = len(df) // chunk_length
    test_size = max(int(num_chunks * test_frac), 1)
    train_size = num_chunks - test_size
    data = (
        pd.DataFrame(df[colname][ticker]).dropna().to_numpy()
        if ticker
        else pd.DataFrame(df[colname]).to_numpy()
    )
    if preprocess:
        scaler = MinMaxScaler(f_range)
        data = scaler.fit_transform(data)

    train_idx, test_idx = train_test_split(np.arange(num_chunks), train_size=train_size)
    index_chunks = [
        np.arange(chunk_length) + chunk_length * i
        for i in range(len(df) // chunk_length)
    ]
    train_data = [data[index_chunks[i]] for i in train_idx]
    train_x, train_y = zip(
        *[(a[:-pred_samples], a[-pred_samples:].squeeze()) for a in train_data]
    )
    test_data = [data[index_chunks[i]] for i in test_idx]
    test_x, test_y = zip(
        *[(a[:-pred_samples], a[-pred_samples:].squeeze()) for a in test_data]
    )
    train_x, train_y, test_x, test_y = (
        [qnp.array(a, requires_grad=False) for a in [train_x, train_y, test_x, test_y]]
        if quantum
        else [np.array(a) for a in [train_x, train_y, test_x, test_y]]
    )

    if val_frac > 0:
        val_size = max(int(len(train_x) * val_frac), 1)
        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=val_size
        )
        return (
            (train_x, train_y, test_x, test_y, val_x, val_y, scaler)
            if preprocess
            else (
                train_x,
                train_y,
                test_x,
                test_y,
                val_x,
                val_y,
            )
        )

    else:
        return (
            (train_x, train_y, test_x, test_y, scaler)
            if preprocess
            else (
                train_x,
                train_y,
                test_x,
                test_y,
            )
        )


def datasets_from_multiindex(
    data,
    seq_length: int = 94,
    pred_samples: int = 1,
    test_frac: float = 0.25,
    val_frac: float = 0.2,
    colname: str = "Perc_Change",
    preprocess: bool = True,
    f_range: Tuple[float, float] = (0.2, 0.8),
    quantum: bool = False,
) -> dict:
    assert isinstance(
        data.columns, pd.MultiIndex
    ), "Pass multiIndexed DataFrame or call `construct_dataset` with ticker=None."
    tickers = data.columns.levels[1]
    datasets = dict.fromkeys(tickers, {})
    for ticker in tickers:
        data_back = construct_dataset(
            data,
            ticker=ticker,
            seq_length=seq_length,
            pred_samples=pred_samples,
            test_frac=test_frac,
            val_frac=val_frac,
            colname=colname,
            preprocess=preprocess,
            f_range=f_range,
            quantum=quantum,
        )
        datasets[ticker]["train_x"] = data_back[0]
        datasets[ticker]["train_y"] = data_back[1]
        datasets[ticker]["test_x"] = data_back[2]
        datasets[ticker]["test_y"] = data_back[3]
        if preprocess:
            assert len(data_back) % 2 == 1
            datasets[ticker]["scaler"] = data_back[-1]
        if len(data_back) > 5:
            datasets[ticker]["val_x"] = data_back[4]
            datasets[ticker]["val_y"] = data_back[5]

    return datasets


if __name__ == "__main__":
    pass
