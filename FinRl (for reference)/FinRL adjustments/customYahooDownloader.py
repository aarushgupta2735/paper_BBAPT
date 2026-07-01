"""Contains methods and classes to collect data from
Yahoo Finance API
"""
# Change at /home/aarushgupta2735/miniconda3/envs/finrl-env/lib/python3.10/site-packages/finrl/meta/preprocessor/yahoodownloader.py
from __future__ import annotations

import pandas as pd
import yfinance as yf


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic, start=self.start_date, end=self.end_date, auto_adjust=True,
            )
            # Flatten multi-level columns
            if isinstance(temp_df.columns, pd.MultiIndex):
                temp_df.columns = temp_df.columns.get_level_values(0)

            # FIX: select and rename columns by name, not position
            temp_df = temp_df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            temp_df = temp_df[["open", "high", "low", "close", "volume"]]
            temp_df["tic"] = tic

            if len(temp_df) > 0:
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1

        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")

        data_df = data_df.reset_index()
        data_df = data_df.rename(columns={"Date": "date", "index": "date"})

        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format
        data_df["date"] = data_df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df
                
    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
