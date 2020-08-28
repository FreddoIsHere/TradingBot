import pandas as pd
from finta import TA
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.techindicators import TechIndicators
from fundamental_data import FundamentalData
from config import key
from sklearn import preprocessing
from tqdm import tqdm

ts = TimeSeries(key, output_format='pandas')
sp = SectorPerformances(key, output_format='pandas')
ti = TechIndicators(key, output_format='pandas')
fd = FundamentalData(key, output_format='pandas')
indicator_periods = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]


def get_daily_data(ticker, compact=True):  # 1 call
    print("\nLoading data for ticker: {}".format(ticker))
    outputsize = 'compact'
    num_data_points = 100
    if not compact:
        outputsize = 'full'
        num_data_points = 1000
    stock, meta_data = ts.get_daily(ticker, outputsize=outputsize)
    stock.columns = ['open', 'high', 'low', 'close', 'volume']
    stock = stock[:num_data_points].iloc[::-1]
    indicators = []
    for period in indicator_periods:
        offset = indicator_periods[-1] - period
        indicators.append(TA.BBANDS(ohlc=stock[offset:], period=period)[period - 1:])  # volatility
        indicators.append(TA.RSI(ohlc=stock[offset:], period=period).to_frame(name='RSI')[period - 1:])  # momentum
        indicators.append(TA.ZLEMA(ohlc=stock[offset:], period=period).to_frame(name='ZLEMA')[period - 1:])  # trend
        indicators.append(TA.EFI(ohlcv=stock[offset:], period=period).to_frame(name='EFI')[period - 1:])  # volume
    indicators.append(TA.ADL(ohlcv=stock).to_frame(name='ADL')[indicator_periods[-1] - 1:])

    df = pd.concat(indicators, axis=1)
    df = pd.concat([stock[indicator_periods[-1] - 1:], df], axis=1)
    return df


def scale_data(df):
    df = df.reset_index()
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    df = scaler.fit_transform(df.to_numpy()[:, 1:])
    return df


def relabel_data(array):
    encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')
    array = encoder.fit_transform(array.reshape(-1, 1))
    return array


def get_sector_performance(sector):  # 1 call
    s_performances, meta_data = sp.get_sector()
    df = s_performances.loc[
        [sector], ['Rank C: Day Performance', 'Rank D: Month Performance', 'Rank E: Month Performance']]
    df.columns = ['5 days', '1 month', '3 months']
    return df


def get_quaterly_data(ticker):
    pass
