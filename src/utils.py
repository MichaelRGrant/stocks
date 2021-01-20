from collections import namedtuple
import datetime
import re
from typing import List, Optional, Tuple, Union

from easydict import EasyDict
import numpy as np
import pandas as pd

from time_series_utils.window import Window


def load_time_series_data(
    model_df_path: str,
    response_horizon: int,
    seq_length: int,
    horizon: int = 1,
    fractional_increase: float = 0.005,
    group_col: str = "symbol",
    train_dates: Tuple[Optional[str], str] = (None, "2019-02-15"),
    valid_dates: Tuple[str, str] = ("2019-02-15", "2019-08-15"),
    test_dates: Tuple[str, str] = ("2019-08-16", "2020-02-18"),
    features: List[str] = [
        "high",
        "low",
        "open",
        "adj_close",
        "volume",
        "macd",
        "psar",
        "roc_20",
        "upper_bb",
        "lower_bb",
        "force_index",
        "stoch_ocs_k_14",
        "stoch_osc_d",
        "obv",
    ],
) -> np.array:
    """
    Load the stocks time series data and return a 3D array for
    use in the tensorflow model.

    Parameters:
    -------
    model_df_path: str
    response_horizon: int
        The number of days to forecast out when creating
        the binary response vector.
    seq_length: int
        How many historical days to use in prediction.
    horizon: int
        Set to 1 if running classification
    fractional_increase: float
        What fraction does the stock need to increase
        to be considered a 1 in the binary response vector.
    group_col: str
        The column to groupby, i.e. the stock symbol column
    train_dates: tuple
        The start and end dates to train on
    valid_dates: tuple
    test_dates: tuple
    features: list
        List of features to use in the model

    Returns:
    -------
    A 3D numpy array in the form (N, timesteps, features)
    """

    model_df = pd.read_feather(model_df_path)

    # add response
    model_df = model_df.groupby(group_col).progress_apply(
        lambda x: create_binary_response(
            x, response_horizon, fractional_increase=fractional_increase
        )
    )

    model_df = model_df.set_index("date")
    model_df = model_df.dropna()

    resp_cols = [f"response_{response_horizon}day_{fractional_increase  * 100}percent"]

    feat_cols = features
    if train_dates[0] is None:
        model_train_df = model_df[: train_dates[1]]
    else:
        model_train_df = model_df[train_dates[0] : train_dates[1]]

    model_val_df = model_df[valid_dates[0] : valid_dates[1]]
    model_test_df = model_df[valid_dates[0] : valid_dates[1]]

    window = Window(
        seq_length=seq_length,
        horizon=horizon,
        feat_cols=feat_cols,
        resp_cols=resp_cols,
        group_col=group_col,
        resp_width=0,
        classification=True,
        predict_current=True,
        scale_by_group=True,
    )

    train = window.make(
        dataset=model_train_df, split=None, scale="minmax", test_set=False
    )
    validation = window.make(
        dataset=model_val_df,
        split=None,
        scale="minmax",
        test_set=True,
        norm_params=window.norm_params,
    )
    test = window.make(
        dataset=model_test_df,
        split=None,
        scale="minmax",
        test_set=True,
        norm_params=window.norm_params,
    )

    train_test_data = EasyDict(
        {
            "train": train,
            "validation": validation,
            "test": test,
            "response": resp_cols[0],
        }
    )

    return train_test_data


def add_response(
    df: pd.DataFrame, horizon, fractional_increase
) -> Tuple[pd.DataFrame, str]:
    """
    Wrapper for the adding the response variable to model.

    Parameters:
    -------
    df: pd.DataFrame
    horizon: int
        the horizon length
    fractional_increase: float
        The fractional increase for the binary response.

    Returns:
    -------
    The augmented dataframe and the response column name.
    """
    df = df.groupby("symbol").progress_apply(
        lambda x: create_binary_response(
            x, horizon, fractional_increase=fractional_increase
        )
    )
    response = [x for x in df.columns if re.search("response", x)][0]
    df = df.set_index("date").dropna()

    return df, response


def split_data_on_date(stock_df: pd.DataFrame, st: str, et: str) -> pd.DataFrame:
    """
    Create a split set for training and testing.

    Parameters:
    -------
    stock_df: pd.DataFrame
    st: str
        Start time to split the data; format %Y-%m-%d
    et: str
        End time to split the data; format %Y-%m-%d

    Returns:
    -------
    pd.DataFrame
        Split dataframe
    """
    stock_df = stock_df.drop(columns=["symbol"]).dropna()
    return stock_df[st:et]


def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    resp_col: Union[str, datetime.date],
    train_st: Union[str, datetime.date],
    train_et: Union[str, datetime.date],
    test_st: Union[str, datetime.date],
    test_et: Union[str, datetime.date],
) -> namedtuple:
    """
    Split the data into training and testing sets on date.

    Parameters:
    -------
    X: pd.DataFrame
        The features dataframe
    y: pd.DataFrame
        The response dataframe
    resp_col: str
        The name of the response column.
    train_st: str
        Date in the format %Y-%m-%d
    train_et: str
        Date in the format %Y-%m-%d
    test_st: str
        Date in the format %Y-%m-%d
    test_et: str
        Date in the format %y-%m-%d

    Returns:
    -------
    namedtuple
        A namedtuple with the traing and testing data split on date.
    """

    train_x = X.groupby("symbol").apply(
        lambda x: split_data_on_date(x, train_st, train_et)
    )
    train_y = y.groupby("symbol").apply(
        lambda x: split_data_on_date(x, train_st, train_et)
    )
    train_y = np.array(train_y.reset_index()[resp_col])

    test_x = X.groupby("symbol").apply(
        lambda x: split_data_on_date(x, test_st, test_et)
    )
    test_y = y.groupby("symbol").apply(
        lambda x: split_data_on_date(x, test_st, test_et)
    )
    test_y = np.array(test_y.reset_index()[resp_col])

    TrainTestData = namedtuple(
        "TrainTestData", ["train_x", "train_y", "test_x", "test_y"]
    )
    train_test_data = TrainTestData._make([train_x, train_y, test_x, test_y])

    return train_test_data


def create_binary_response(
    df: pd.DataFrame, ndays: int, fractional_increase: Optional[float] = None
) -> pd.DataFrame:
    """
    Create the binary response vector for modeling.

    Parameters:
    -------
    df: pd.DataFrame
        Dataframe with stock data.
    ndays: int
        Number of days in the future to calculate the `fractional_increase`.
    fractional_increase: float; (0, 1]
        If the response should only return a 1 if the stock went
        up by `fractional_increase` or more.
    Returns:
    -------
    """
    if fractional_increase:
        response = f"response_{ndays}day_{fractional_increase * 100}percent"
        df[response] = np.where(
            df["close"].shift(-ndays) / df["open"] - 1 >= fractional_increase, 1, 0
        )
    else:
        response = f"response_{ndays}"
        df[response] = np.where(df["close"].shift(-ndays) > df["open"], 1, 0)

    return df


def return_symbols_with_enough_time(df: pd.DataFrame, nyears: int) -> list:
    """
    Return a list of stocks that contain time up to the number of years.

    Parameters:
    -------
    stocks_df: pd.DataFrame
        The dataframe with stock and technical features.
    nyears: int
        The year cutoff.

    Returns:
    -------
    list
        A list of the stock symbosl that meet cutoff criteria.
    """

    return (
        df.groupby("symbol")
        .size()
        .reset_index(name="size")
        .query(f"size > {nyears * 365}")["symbol"]
    )


def series_to_supervised(
    df: pd.DataFrame,
    features: List[str],
    group_name: str,
    response_name: str,
    lag: int = 1,
    horizon: int = 1,
    dropnan: bool = True,
):
    """
    Frame a time series as a supervised learning dataset.
    Parameters:
    -------
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    cols, names = [], []
    # input sequence (t-n, ... t-1)
    for i in range(lag, 0, -1):
        for col in features:
            cols.append(df[col].shift(i))
            names += [f"{col}(t-{i})"]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, horizon):
        for col in features:
            cols.append(df[col].shift(-i))
            if i == 0:
                names += [f"{col}(t)"]
            else:
                names += [f"{col}(t+{i}"]
    # put it all together

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg["symbol"] = group_name
    agg["date"] = df["date"]
    agg[response_name] = df[response_name]
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
