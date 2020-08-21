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


def get_daily_data(ticker):  # 4 calls
    stock, meta_data = ts.get_daily(ticker, outputsize='compact')
    intra_day_variation_pos = stock['2. high'] / stock['4. close'] - 1
    intra_day_variation_pos.columns = ['Var_pos']
    intra_day_variation_neg = stock['3. low'] / stock['4. close'] - 1
    intra_day_variation_neg.columns = ['Var_neg']
    returns = stock['4. close'] / stock['4. close'].shift(1) - 1
    returns.columns = ['Returns']
    bbands, _ = ti.get_bbands(ticker, interval='daily', time_period=7, series_type='close')  # volatility
    rsi, _ = ti.get_rsi(ticker, interval='daily', time_period=7, series_type='close')  # momentum
    obv, _ = ti.get_obv(ticker, interval='daily')  # volume
    df = pd.concat([bbands.head(100).iloc[::-1], returns, intra_day_variation_pos, intra_day_variation_neg,
                    rsi.tail(100).iloc[::-1], obv.tail(100).iloc[::-1]], axis=1)[:-1]
    df.columns = ['UBB', 'MBB', 'LBB', 'Return', 'Pos_var', 'Neg_var', 'RSI', 'OBV']
    return df


def get_sector_performance(sector):  # 1 call
    s_performances, meta_data = sp.get_sector()
    df = s_performances.loc[
        [sector], ['Rank C: Day Performance', 'Rank D: Month Performance', 'Rank E: Month Performance']]
    df.columns = ['5 days', '1 month', '3 months']
    return df

def get_quaterly_data(ticker):
    pass
