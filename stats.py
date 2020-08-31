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
indicator_periods = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


def get_daily_data(ticker, compact=True):  # 1 call
    print("\nLoading data for ticker: {}".format(ticker))
    outputsize = 'compact'
    num_data_points = 100
    if not compact:
        outputsize = 'full'
        num_data_points = 1000 + indicator_periods[-1]*2
    stock, meta_data = ts.get_daily(ticker, outputsize=outputsize)
    stock.columns = ['open', 'high', 'low', 'close', 'volume']
    stock = stock[:num_data_points].iloc[::-1]
    indicators = []
    for period in indicator_periods:
        indicators.append(TA.TEMA(ohlc=stock, period=period).to_frame(name='TEMA'))
        indicators.append(TA.BBANDS(ohlc=stock, period=period))  # volatility
        indicators.append(TA.MI(ohlc=stock, period=period).to_frame(name='MI'))  # volatility
        indicators.append(TA.ATR(ohlc=stock, period=period).to_frame(name='ATR'))  # volatility
        indicators.append(TA.RSI(ohlc=stock, period=period).to_frame(name='RSI'))  # momentum
        indicators.append(TA.MFI(ohlc=stock, period=period).to_frame(name='MFI'))  # momentum
        indicators.append(TA.WILLIAMS(ohlc=stock, period=period).to_frame(name='WILLIAMS'))  # momentum
        indicators.append(TA.ZLEMA(ohlc=stock, period=period).to_frame(name='ZLEMA'))  # trend
        indicators.append(TA.WMA(ohlc=stock, period=period).to_frame(name='WMA'))  # trend
        indicators.append(TA.HMA(ohlc=stock, period=period).to_frame(name='HMA'))  # trend x
        indicators.append(TA.VAMA(ohlcv=stock, period=period).to_frame(name='VAMA'))  # volume x
        indicators.append(TA.EFI(ohlcv=stock, period=period).to_frame(name='EFI'))  # volume
        indicators.append(TA.EMV(ohlcv=stock, period=period).to_frame(name='EMV'))  # volume
    df = pd.concat(indicators, axis=1)[indicator_periods[-1]*2:]
    stock = stock[indicator_periods[-1]*2:]
    return stock, df


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
