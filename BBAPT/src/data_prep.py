import pandas as pd
import yfinance as yf
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator
#2019-02-06 2022-06-07 2022-06-08 2023-01-03

def add_technical_indicators(df,ticker):
    df = df.copy()
    macd_int = MACD(close=df[('Close',ticker)], window_fast=12, window_slow=26, window_sign=9)
    df[('MACD',ticker)] = macd_int.macd()

    # Relative Strength Index (RSI)
    df[('RSI_14',ticker)] = RSIIndicator(close=df[('Close',ticker)], window=14).rsi()

    # Commodity Channel Index (CCI)
    df[('CCI_20',ticker)] = CCIIndicator(high=df[('High',ticker)], low=df[('Low',ticker)], close=df[('Close',ticker)], window=20).cci()

    # Average Directional Index (ADX)    
    #adx_int = ADXIndicator(high=df[('High',ticker)], low=df[('Low',ticker)], close=df[('Close',ticker)], window=14)
    #df[('ADX_14',ticker)] = adx_int.adx()

    # Simple Moving Average (SMA)
    df[('SMA_20',ticker)] = SMAIndicator(close=df[('Close',ticker)], window=20).sma_indicator()

    # Exponential Moving Average (EMA)
    df[('EMA_20',ticker)] = EMAIndicator(close=df[('Close',ticker)], window=20).ema_indicator()
    return df

def data_prep(starting_date, ending_date,ticker_list):
    data = yf.download("BTC-USD ETH-USD LTC-USD LINK-USD BCH-USD UNI-USD XLM-USD FIL-USD BNB-USD SOL-USD XRP-USD ADA-USD SHIB-USD TON-USD DOGE-USD AVAX-USD TRX-USD DOT-USD MATIC-USD ETC-USD", start=starting_date, end=ending_date)
    for ticker in ticker_list:
        data[('Returns',ticker)] = data[('Close',ticker)] - data[('Close',ticker)].iloc[0] #Change in price from the first day of the data #TODO
    data.columns.names = ['Attributes','Ticker']
    for ticker in ticker_list:
        data = add_technical_indicators(data,ticker)
    data = data.stack(level = 'Ticker').reset_index()
    data['std_dev'] = data.groupby('Ticker')['Returns'].expanding().std().reset_index(level=0, drop=True)
    data = data.sort_values(by = ['Date','Ticker'])
    data.rename(columns = {'Date':'ds', 'Returns':'y','Ticker':'unique_id'}, inplace = True)
    data = data[data['ds']!=data['ds'].min()]
    nan_dates = data.loc[data.isna().any(axis=1),'ds'].unique()
    data = data[~data['ds'].isin(nan_dates)]
    return data, len(nan_dates)

def data_for_timesnet(data):
    return data[['ds', 'y', 'unique_id']]

