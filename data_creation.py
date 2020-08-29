from tqdm import tqdm
from stats import get_daily_data, scale_data, relabel_data
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import pickle
import torch
import time
import sys
import os
from config import training_indices

current_folder = os.getcwd()


class DataCreator:
    def __init__(self, batch_size, path=current_folder):
        self.path = path
        self.batch_size = batch_size

    def provide_training_stock(self):
        stocks = []
        signals = []
        with open(self.path + '/sp100_stocks.pkl', 'rb') as infile_1:
            with open(self.path + '/sp100_signals.pkl', 'rb') as infile_2:
                for _ in range(len(training_indices)):
                    stocks.append(pickle.load(infile_1))
                    signals.append(pickle.load(infile_2))
        stocks = np.vstack(stocks)
        signals = np.hstack(signals)
        n = signals.size
        class_weights = torch.tensor([n/sum(1 if x == 0 else 0 for x in signals), n/sum(1 if x == 1 else 0 for x in signals), n/sum(1 if x == 2 else 0 for x in signals)]).float()
        class_weights = class_weights/class_weights.sum()
        weights = class_weights[signals]
        train_set = TensorDataset(torch.from_numpy(stocks), torch.from_numpy(signals))
        sampler = WeightedRandomSampler(weights=weights, num_samples=n, replacement=True)
        return DataLoader(train_set, batch_size=self.batch_size, sampler=sampler), class_weights

    def create_data(self, tickers, window_size=11):
        with open(self.path + '/sp100_stocks.pkl', 'wb') as outfile_1:
            with open(self.path + '/sp100_signals.pkl', 'wb') as outfile_2:
                for t in tickers:
                    start_time = time.time()
                    stock = get_daily_data(t, False)
                    signals = self.create_labels(stock, window_size=window_size)[window_size:]
                    stock = stock[window_size:]
                    pickle.dump(scale_data(stock), outfile_1, pickle.HIGHEST_PROTOCOL)
                    pickle.dump(signals, outfile_2, pickle.HIGHEST_PROTOCOL)
                    elapsed_time = time.time()
                    time_to_sleep = int(13 - (elapsed_time - start_time))
                    for i in range(time_to_sleep, 0, -1):  # only 5 api calls per minute allowed
                        sys.stdout.write("\r")
                        sys.stdout.write("Waiting time for next API call: {:2d}s".format(i))
                        sys.stdout.flush()
                        time.sleep(1)
        print("\nAll data retrieved!")

    def create_labels(self, df, col_name='close', window_size=11):
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


if __name__ == "__main__":
    try:
        print("Deleting old training data!")
        os.remove(current_folder + '/sp100_stocks.pkl')
        os.remove(current_folder + '/sp100_signals.pkl')
    except:
        print("Data retrieval started!")
    creator = DataCreator(128)
    creator.create_data(training_indices)