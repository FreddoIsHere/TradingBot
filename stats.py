import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

def get_stock_data(ticker, start, end):
    """
    String:param ticker: Stock ticker
    DateTime:param start: Start date of reading
    DateTime:param end: End date of reading
    :return:
    """
    df = web.DataReader(ticker, 'yahoo', start, end)
    closing_vals = df['Adj Close']
    moving_average = closing_vals.rolling(window=14).mean()
    returns = closing_vals / closing_vals.shift(1) - 1