import itertools
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import talib
from tqdm import tqdm_notebook


def add_all_technical_features_from_paper(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all the features to the data.

    Parameters:
    -------
    grouped_stock_df: pd.DataFrame
        A single stock dataframe

    Returns:
    -------
    pd.DataFrame
        Dataframe with all the features added.
    """
    stock_df_augmented = (
        macd(stock_df)
        .pipe(parabolic_sar)
        .pipe(calculate_roc, 20)
        .pipe(bollinger_bands)
        .pipe(force_index, 10)
        .pipe(calculate_stochastic)
        .pipe(add_on_balance_volume)
        .pipe(ema, ["close"], [5, 10, 20, 50])
        .pipe(calc_crossover, ma_windows=[5, 10, 20, 50], ma_type=["ema"])
    )
    return stock_df_augmented


def add_all_technical_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all the features to the data.

    Parameters:
    -------
    grouped_stock_df: pd.DataFrame
        A single stock dataframe

    Returns:
    -------
    pd.DataFrame
        Dataframe with all the features added.
    """
    stock_df_augmented = (
        cci(stock_df, 20)
        .pipe(macd)
        .pipe(parabolic_sar)
        .pipe(evm, 20)
        .pipe(calculate_roc, 20)
        .pipe(bollinger_bands)
        .pipe(force_index, 10)
        .pipe(calculate_rsi)
        .pipe(calculate_stochastic)
        .pipe(add_on_balance_volume)
        .pipe(sma, ["close"], [5, 10, 20, 50])
        .pipe(ema, ["close"], [5, 10, 20, 50])
        .pipe(dema, ["close"], [5, 10, 20, 50])
        .pipe(calc_crossover, ma_windows=[5, 10, 20, 50])
#         .pipe(get_slopes, ndays=5, cols=["dema_close_5"])
    )
#     stock_df_augmented = ema(
#         stock_df_augmented,
#         [
#             col
#             for col in stock_df_augmented.columns
#             if re.search("cross_above|cross_below", col)
#         ],
#         [10, 20],
#     )

    return stock_df_augmented


def get_slopes(df: pd.DataFrame, ndays: int, cols: List[int]) -> pd.DataFrame:
    """
    Using simple linear regression to calculate the slopes for the
    number of `ndays` back for each column in `cols`.
    """
    lin_reg = LinearRegression()
    X = np.arange(5).reshape(-1, 1)
    for col in cols:

        slopes = [
            lin_reg.fit(
                X, MinMaxScaler().fit_transform(df.iloc[i : i + ndays][[col]]).ravel()
            ).coef_[0]
            for i in tqdm_notebook(
                range(df.shape[0] - (ndays - 1)), total=df.shape[0] - ndays, leave=False
            )
        ]

    df = df.join(
        pd.Series(slopes, index=range(ndays - 1, df.shape[0]), name=f"slopes_{col}")
    )

    return df


def sma(grp_df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Calculate the simple moving average.

    Parameters:
    -------
    grp_df: pd.DataFrame
        The grouped dataframe.
    col: str
    window: list
        List of windows to take simple moving average.
    """
    for window in windows:
        for col in cols:
            grp_df[f"sma_{col}_{window}"] = grp_df[col].rolling(window=window).mean()
    return grp_df


def ema(grp_df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Calculate the exponential moving average.

    Parameters:
    -------
    grp_df: pd.DataFrame
        The grouped dataframe.
    col: str
    window: list
        List of windows to take simple moving average.
    """
    for window in windows:
        for col in cols:
            grp_df[f"ema_{col}_{window}"] = (
                grp_df[col].ewm(span=window, adjust=True).mean()
            )
    return grp_df


def dema(df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Calculate the double exponential moving average.

    Parameters:
    -------
    grp_df: pd.DataFrame
        The grouped dataframe.
    col: str
    window: list
        List of windows to take simple moving average.
    """
    for window in windows:
        for col in cols:
            ema = df[col].ewm(span=window, adjust=True).mean()
            ema_of_ema = ema.ewm(span=window).mean()
            df[f"dema_{col}_{window}"] = 2 * ema - ema_of_ema
    return df


def calc_crossover(
    df: pd.DataFrame,
    ma_type: Tuple = ("sma", "ema", "dema"),
    col: str = "close",
    ma_windows: Optional[List] = None,
) -> pd.DataFrame:
    """
    Calculate if two moving averages cross each other.

    Parameters:
    -------
    grp_df: pd.DataFrame
        The grouped dataframe.
    ma_type: tuple
        Each moving average type in data.
    col: str
        The column to calculate the moving average; default `close`
    ma_windows: Union[list, None]
        Choose which moving averages to calculate cross over.
    """
    for ma in ma_type:
        ma_grp_cols = df.columns[df.columns.str.contains(f"^{ma}")]
        # If `None` make crossovers for all moving averages.
        if ma_windows is None:
            ma_windows = [int(re.search("[^_]+$", x).group(0)) for x in ma_grp_cols]

        #         if ma == "dema":
        #             ma_cross_combos = [(5, 10, 20), (10, 50), (20,50)]
        #         else:

        ma_cross_combos = list(itertools.combinations(ma_windows, 2))

        for ma_cross in ma_cross_combos:
            df[f"{ma}_{ma_cross[0]}_cross_above_{ma}_{ma_cross[1]}"] = (
                (
                    df[f"{ma}_{col}_{ma_cross[0]}"].shift(1)
                    < df[f"{ma}_{col}_{ma_cross[1]}"]
                )
                & (df[f"{ma}_{col}_{ma_cross[0]}"] > df[f"{ma}_{col}_{ma_cross[1]}"])
            ) * 1
            df[f"{ma}_{ma_cross[0]}_cross_below_{ma}_{ma_cross[1]}"] = (
                (
                    df[f"{ma}_{col}_{ma_cross[0]}"].shift(1)
                    > df[f"{ma}_{col}_{ma_cross[1]}"]
                )
                & (df[f"{ma}_{col}_{ma_cross[0]}"] < df[f"{ma}_{col}_{ma_cross[1]}"])
            ) * 1

    return df


def cci(df: pd.DataFrame, ndays: int) -> pd.DataFrame:
    cci = pd.Series(
        talib.CCI(df["high"], df["low"], df["close"], timeperiod=ndays),
        name=f"cci_{ndays}",
    )
    df = df.join(cci)
    return df


# deprecated
# def cci(df: pd.DataFrame, ndays: int) -> pd.DataFrame:
#     """
#     Calculate commodity channel index.
#
#     Parameters:
#     -------
#     df: pd.DataFrame
#         Dataframe with stock data.
#     ndays: int
#         The number of days back to calculate commodity channel index.
#
#     Returns:
#     -------
#     pd.DataFrame
#         Dataframe with commodity channel index.
#     """
#     tp = (df["high"] + df["low"] + df["close"]) / 3
#     cci = pd.Series(
#         (tp - tp.rolling(ndays).mean()) / (0.015 * tp.rolling(ndays).std()),
#         name=f"cci_{ndays}",
#     )
#     df = df.join(cci, )
#     return df


# Ease of Movement
def evm(df: pd.DataFrame, ndays: int) -> pd.DataFrame:
    """
    Calculate ease of movement.

    Parameters:
    -------
    df: pd.DataFrame
        Dataframe with stock data.
    ndays: int
        The number of days back to ease of movement index.

    Returns:
    -------
    pd.DataFrame
        Dataframe with ease of movement index.
    """
    distance_moved = ((df["high"] + df["low"]) / 2) - (
        (df["high"].shift(1) + df["low"].shift(1)) / 2
    )
    box_ratio = (df["volume"] / 100000000) / (df["high"] - df["low"])
    evm = distance_moved / box_ratio
    evm_ma = pd.Series(evm.rolling(ndays).mean(), name=f"evm_{ndays}")
    df = df.join(evm_ma)
    return df


# Compute the Bollinger Bands
def bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate bollinger bands.

    Parameters:
    -------
    df: pd.DataFrame
        Dataframe with single stock data.
    window: int
        The moving average window to calculate bollinger bands.

    Returns:
    -------
    pd.DataFrame
        Dataframe with bollinger bands added.
    """
    ma = df["close"].rolling(window).mean()
    ma_sd = df["close"].rolling(window).std()
    df["upper_bb"] = ma + (2 * ma_sd)
    df["lower_bb"] = ma - (2 * ma_sd)
    df["bb_diff"] = df["upper_bb"] - df["lower_bb"]
    return df


# Force Index
def force_index(df: pd.DataFrame, ndays: int) -> pd.DataFrame:
    """
    Calculate force index.

    Parameters:
    -------
    df: pd.Dataframe
        Dataframe of stock data.
    ndays: int
        Number of days back to calculate index by.

    Returns:
    ------
    Dataframe with force index added.
    """
    fi = pd.Series(df["close"].diff(ndays) * df["volume"], name="force_index")
    df = df.join(fi)
    return df


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate relative strength index.

    Parameters:
    -------
    grp_df: pd.DataFrame

    Returns:
    -------
    Dataframe with RSI added.
    """
    close = df["close"]
    # Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up = up.ewm(span=window).mean()
    roll_down = down.abs().ewm(span=window).mean()

    # Calculate the RSI based on EWMA
    rsi = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rsi))
    rsi = pd.Series(rsi, name=f"rsi_{window}")
    df = df.join(rsi)

    return df


def calculate_roc(df: pd.DataFrame, ndays: int) -> pd.DataFrame:
    """
    Calculate rate of change.

    Parameters:
    -------
    df: pd.DataFrame
        Dataframe with stock data.
    ndays: int
        Number of days back to calculate rate of change.

    Returns:
    -------
    pd.DataFrame
        Dataframe with rate of change added.
    """

    roc = pd.Series(
        (df["close"] - df["close"].shift(ndays) / (df["close"].shift(ndays))) * 100,
        name=f"roc_{ndays}",
    )
    df = df.join(roc)
    return df


def calculate_stochastic(df: pd.DataFrame, ndays: int = 14) -> pd.DataFrame:
    """
    Calculate the stochastic oscillator.

    Parameters:
    -------
    df: pd.DataFrame
        Dataframe with stock data.
    ndays: int
        Number of days to calculate stochastic oscillator.

    Returns:
    -------
    pd.DataFrame
        Dataframe with stochastic oscillator K and D added.
    """
    stoch_osc_k = pd.Series(
        (df["close"] - df["close"].rolling(ndays).min())
        / (df["close"].rolling(ndays).max() - df["close"].rolling(ndays).min()),
        name=f"stoch_ocs_k_{ndays}",
    )
    stock_osc_d = pd.Series(stoch_osc_k.rolling(3).mean(), name=f"stoch_osc_d")

    df = df.join(stoch_osc_k)
    df = df.join(stock_osc_d)

    return df


def add_on_balance_volume(
    df: pd.DataFrame,
):
    """
    Calculate the on balance volume.
    Indicator type: Leading
    """
    obv = pd.Series(talib.OBV(df["close"], df["volume"]), name="obv")
    df = df.join(obv)
    return df


def parabolic_sar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the parabolic stop and reverse signal.

    Parameters:
    -------
    df: pd.Dataframe
        A single stock dataframe

    Returns:
    -------
    df: pd.DataFrame
    """
    psar = pd.Series(talib.SAR(df["high"], df["low"]), name="psar")
    df = df.join(psar)
    return df


def macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the moving average convergence or divergence indicator.

    Parameters:
    -------
    df: pd.DataFrame
        Datframe for a single stock.

    Returns:
    -------
    df: pd.DataFrame
    """
    macd, _, _ = talib.MACD(df["close"])
    macd = pd.Series(macd, name="macd")
    df = df.join(macd)
    return df


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the date/time features to the model.

    Parameters:
    -------
    df: pd.DataFrame
        Dataframe with a datetime features `date` for each
        days stock information.

    Returns:
    -------
    df: pd.DataFrame
    """
    dayofmonth = pd.Series(df["date"].dt.day, name="dayofmonth")
    dayofweek = pd.Series(df["date"].dt.dayofweek, name="dayofweek")
    dayofyear = pd.Series(df["date"].dt.dayofyear, name="dayofyear")

    week = pd.Series(df["date"].dt.week, name="week")
    month = pd.Series(df["date"].dt.month, name="month")
    year = pd.Series(df["date"].dt.year, name="year")
    df = df.join(
        pd.concat([year, month, dayofmonth, dayofweek, dayofyear, week], axis=1)
    )

    return df
