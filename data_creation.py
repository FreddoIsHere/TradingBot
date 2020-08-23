from tqdm import tqdm
from stats import get_daily_data, scale_data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import torch
import time
import sys
import os

current_folder = os.getcwd()


class DataCreator:
    def __init__(self, path=current_folder):
        self.path = path

    def provide_training_stock(self):
        stock = pd.read_pickle(self.path + '/stocks')
        signals = pd.read_pickle(self.path + '/signals')
        train_X = torch.from_numpy(stock.values)
        train_Y = torch.from_numpy(signals.values)
        train_set = TensorDataset(train_X, train_Y)
        return DataLoader(train_set, shuffle=False, batch_size=16)

    def create_data(self, tickers):
        for t in tickers:
            start_time = time.time()
            stock = get_daily_data(t, False)
            signals = self.create_labels(stock)
            scale_data(stock).to_pickle(self.path + '/stocks')
            signals.to_pickle(self.path + '/signals')
            elapsed_time = time.time()
            time_to_sleep = int(13 - (elapsed_time - start_time))
            for i in range(time_to_sleep, 0, -1):  # only 5 api calls per minute allowed
                sys.stdout.write("\r")
                sys.stdout.write("Waiting time for next API call: {:2d}s".format(i))
                sys.stdout.flush()
                time.sleep(1)
        print("\nAll data retrieved!")

    def create_labels(self, df, col_name='close', window_size=21):
        """
        Label code : BUY => 1, SELL => 0, HOLD => 2
        """
        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        labels[:window_size] = 2
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
        df = pd.DataFrame(data=labels, columns=['Signal'])
        return df
