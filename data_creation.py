import numpy as np
import pandas as pd
from tqdm import tqdm
from stats import get_daily_data
import time
import sys


class DataCreator:
    def __init__(self, tickers):
        self.tickers = tickers
        data, labels = self.create_data(tickers)
        self.stocks = pd.concat(data, keys=tickers)

    def create_data(self, tickers):
        stocks = []
        labels = []
        for t in tickers:
            start_time = time.time()
            stock = get_daily_data(t, False)
            label = self.create_labels(stock)
            stocks.append(stock)
            labels.append(label)
            elapsed_time = time.time()
            time_to_sleep = int(61 - (elapsed_time - start_time))
            for i in range(time_to_sleep, 0, -1): # only 5 api calls per minute allowed
                sys.stdout.write("\r")
                sys.stdout.write("Waiting time for next API call: {:2d}s".format(i))
                sys.stdout.flush()
                time.sleep(1)

        return stocks, labels

    def create_labels(self, df, col_name='Close', window_size=10):
        """
        Label code : BUY => 1, SELL => 0, HOLD => 2
        """
        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)

        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = (window_begin + window_end) / 2

                min_ = np.inf
                min_index = -1
                max_ = -np.inf
                max_index = -1
                for i in range(window_begin, window_end + 1):
                    price = df.iloc[i][col_name]
                    if price < min_:
                        min_ = price
                        min_index = i
                    if price > max_:
                        max_ = price
                        max_index = i

                if max_index == window_middle:
                    labels[row_counter] = 0
                elif min_index == window_middle:
                    labels[row_counter] = 1
                else:
                    labels[row_counter] = 2

            row_counter = row_counter + 1
            pbar.update(1)

        pbar.close()
        return labels


dc = DataCreator(['MSFT'])
