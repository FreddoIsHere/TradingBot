import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.techindicators import TechIndicators
from fundamental_data import FundamentalData
from config import key

ts = TimeSeries(key, output_format='pandas')
sp = SectorPerformances(key, output_format='pandas')
ti = TechIndicators(key, output_format='pandas')
fd = FundamentalData(key, output_format='pandas')


def get_daily_data(ticker):
    stock, meta_data = ts.get_daily(ticker, outputsize='compact')
    closing_vals = stock['4. close']
    moving_avg = closing_vals.rolling(window=7, min_periods=1).mean()
    returns = closing_vals / closing_vals.shift(1) - 1
    volume = stock['5. volume']
    return pd.concat([moving_avg, returns, volume], keys=['Avg', 'Returns', 'Volume'], axis=1)[1:]

def get_quarterly_data(ticker):
    stock, meta_data = sp.get_sector

print(fd.get_income_statement('MSFT')[0])
