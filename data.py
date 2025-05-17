# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
from typing import Union, Tuple

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def parse_json_dir(dir_path: Union[str, os.PathLike]):
    out_df = pd.DataFrame()
    for fn in os.listdir(dir_path):
        df = pd.read_json(os.path.join(dir_path, fn))
        out_df = pd.concat([out_df, df])
    out_df = out_df.sort_values("Date", ascending=True).reset_index(drop=True)

    out_df["Prec_Change"] = out_df.Adj_Close.diff() / out_df.Adj_Close
    return out_df.dropna()


def plot_df(df: pd.DataFrame):
    plt.plot_date(df.Date, df.Adj_Close, ".k")
    assert len(df.Symbol.unique()) == 1, "Only do one stock at a time."
    plt.title("$" + df.Symbol[0])
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.show()


# we are going to in=gnore after-hours trading and say it is uniformly sampled in trading-days
def construct_dataset(
    df,
    chunk_length: int = 95,
    test_frac: float = 0.25,
    val_frac: float = 0.2,
    colname: str = "Adj_Close",
    preprocess: bool = True,
    f_range: Tuple[float, float] = (0.2, 0.8),
):
    if test_frac > 1:
        test_frac = test_frac / len(df)
        print(
            f"test_frac > 1 passed. Assuming it is number of test samples. Set test_frac to {test_frac}."
        )

    # want to test in a random selection of chunks
    num_chunks = len(df) // chunk_length
    test_size = max(int(num_chunks * test_frac), 1)
    train_size = num_chunks - test_size
    data = pd.DataFrame(df[colname])
    if preprocess:
        data = MinMaxScaler(f_range).fit_transform(data)

    train_idx, test_idx = train_test_split(np.arange(num_chunks), train_size=train_size)
    index_chunks = [
        np.arange(chunk_length) + chunk_length * i
        for i in range(len(df) // chunk_length)
    ]
    train_data = [data[index_chunks[i]] for i in train_idx]
    train_x, train_y = zip(*[(a[:-1], a[-1]) for a in train_data])
    test_data = [data[index_chunks[i]] for i in test_idx]
    test_x, test_y = zip(*[(a[:-1], a[-1]) for a in test_data])
    train_x, train_y, test_x, test_y = (
        np.array(train_x),
        np.array(train_y),
        np.array(test_x),
        np.array(test_y),
    )
    if val_frac > 0:
        val_size = max(int(len(train_x) * val_frac), 1)
        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=val_size
        )
        return train_x, train_y, test_x, test_y, val_x, val_y
    else:
        return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    pass
