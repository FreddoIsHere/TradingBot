import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.techindicators import TechIndicators
from fundamental_data import FundamentalData
from config import key
from sklearn import preprocessing

ts = TimeSeries(key, output_format='pandas')
sp = SectorPerformances(key, output_format='pandas')
ti = TechIndicators(key, output_format='pandas')
fd = FundamentalData(key, output_format='pandas')


def get_daily_data(ticker, compact=True):  # 4 calls
    print("Loading data for ticker: {}".format(ticker))
    outputsize = 'compact'
    num_data_points = 100
    if not compact:
        outputsize = 'full'
        num_data_points = 1000
    stock, meta_data = ts.get_daily(ticker, outputsize=outputsize)
    bbands, _ = ti.get_bbands(ticker, interval='daily', time_period=5, series_type='close')  # volatility
    rsi, _ = ti.get_rsi(ticker, interval='daily', time_period=5, series_type='close')  # momentum
    obv, _ = ti.get_obv(ticker, interval='daily')  # volume
    df = pd.concat([stock[['2. high', '3. low', '4. close', '5. volume']].head(num_data_points),
                    bbands.head(num_data_points).iloc[::-1], rsi.tail(num_data_points).iloc[::-1],
                    obv.tail(num_data_points).iloc[::-1]], axis=1)[:-1]
    df.columns = ['High', 'Low', 'Close', 'Volume', 'UBB', 'MBB', 'LBB', 'RSI', 'OBV']
    return df


def scale_data(df):
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    df = scaler.fit_transform(df)
    return pd.DataFrame(data=df, index=df[0:, 0], columns=['High', 'Low', 'Close', 'Volume', 'UBB', 'MBB', 'LBB', 'RSI', 'OBV'])


def get_sector_performance(sector):  # 1 call
    s_performances, meta_data = sp.get_sector()
    df = s_performances.loc[
        [sector], ['Rank C: Day Performance', 'Rank D: Month Performance', 'Rank E: Month Performance']]
    df.columns = ['5 days', '1 month', '3 months']
    return df


def get_quaterly_data(ticker):
    pass
