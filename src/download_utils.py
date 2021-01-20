import datetime
from typing import Union

import pandas as pd
import pandas_datareader as pdr
from tqdm import tqdm_notebook


def download_pdr(
    stock_symbols: list,
    st: Union[str, datetime.date] = "1970-01-01",
    et: Union[str, datetime.date] = datetime.date.today(),
) -> pd.DataFrame:

    """
    Scrape stocks using pandas datareader.

    Parameters:
    -------
    stock_symbols: list
        list of stock symbols.
    st: str
        Start time as %Y%m%d
    et: str
        End time as %Y%m%d

    Returns:
    -------
    stocks_df: pd.DataFrame
    """
    stocks_list = []
    for symbol in tqdm_notebook(
        stock_symbols, desc="download stocks", total=len(stock_symbols)
    ):
        try:
            stocks_list.append(
                pdr.DataReader(symbol, start=st, end=et, data_source="yahoo").assign(
                    symbol=symbol
                )
            )
        except:
            # Some stocks in the list of symbols return an error.
            # They may be companies that no longer exist.
            continue
    stocks_df = pd.concat(stocks_list).reset_index()
    stocks_df.columns = [col.lower().replace(" ", "_") for col in stocks_df.columns]
    stocks_df["adj_close"] = stocks_df["adj_close"].astype(float)

    return stocks_df
