from tqdm import tqdm
from stats import get_daily_data, scale_data, relabel_data
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import pickle
import torch
import time
import sys
import os
import argparse
from config import training_indices, testing_indices
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
        class_weights = torch.tensor([n/sum(1 if x == 0 else 0 for x in signals), n/sum(1 if x == 1 else 0 for x in signals), n/sum(1 if x == 2 else 0 for x in signals), n/sum(1 if x == 3 else 0 for x in signals)]).float()
        class_weights = class_weights/class_weights.sum()
        weights = class_weights[signals]
        stocks = torch.from_numpy(stocks)
        signals = torch.from_numpy(relabel_data(signals))
        train_set = TensorDataset(stocks.reshape(-1, 15, 15), signals)
        sampler = WeightedRandomSampler(weights=weights, num_samples=n, replacement=True)
        return DataLoader(train_set, batch_size=self.batch_size, sampler=sampler), n

    def provide_testing_stock(self):
        stocks = []
        signals = []
        with open(self.path + '/test_stocks.pkl', 'rb') as infile_1:
            with open(self.path + '/test_signals.pkl', 'rb') as infile_2:
                for _ in range(len(testing_indices)):
                    stocks.append(pickle.load(infile_1))
                    signals.append(pickle.load(infile_2))
        stocks = np.vstack(stocks)
        signals = np.hstack(signals)
        stocks = torch.from_numpy(stocks)
        signals = torch.from_numpy(signals)
        test_set = TensorDataset(stocks.reshape(-1, 15, 15), signals)
        return DataLoader(test_set, batch_size=self.batch_size, shuffle=True)

    def create_data(self, tickers, stock_path, label_path, window_size=11):
        with open(stock_path, 'ab') as outfile_1:
            with open(label_path, 'ab') as outfile_2:
                for t in tickers:
                    start_time = time.time()
                    stock, data = get_daily_data(t, False)
                    signals, shortening = self.create_labels(stock, window_size=window_size)
                    data = data[1:shortening]
                    pickle.dump(scale_data(data), outfile_1, pickle.HIGHEST_PROTOCOL)
                    pickle.dump(signals, outfile_2, pickle.HIGHEST_PROTOCOL)
                    elapsed_time = time.time()
                    time_to_sleep = int(13 - (elapsed_time - start_time))
                    for i in range(time_to_sleep, 0, -1):  # only 5 api calls per minute allowed
                        sys.stdout.write("\r")
                        sys.stdout.write("Waiting time for next API call: {:2d}s".format(i))
                        sys.stdout.flush()
                        time.sleep(1)
        print("\nAll data retrieved!")

    def display_annotated_graphs(self, ticker, window_size=11):
        imperatives = ['SELL', 'BUY', 'HOLD', 'STOP']
        stock, data = get_daily_data(ticker, True)
        signals, shortening = self.create_labels(stock, window_size=window_size)
        stock = stock[1:shortening]['close']
        plt.plot(stock)
        for l in range(shortening-1):
            plt.annotate(xy=(l, stock[l]), text=imperatives[signals[l]])
        plt.show()

    def create_labels(self, df, col_name='close', window_size=11):
        """
        Label code : SELL => 0, BUY => 1, HOLD => 2, STOP => 3
        """
        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)

        last_index = 0
        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = int(np.ceil((window_begin + window_end) / 2))

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
                    labels[window_middle] = 0
                    if df.iloc[window_middle][col_name] >= df.iloc[last_index][col_name]:
                        labels[last_index + 1: window_middle] = 2
                    else:
                        labels[last_index + 1: window_middle] = 3
                    last_index = window_middle
                elif min_index == window_middle:
                    labels[window_middle] = 1
                    if df.iloc[window_middle][col_name] >= df.iloc[last_index][col_name]:
                        labels[last_index + 1: window_middle] = 2
                    else:
                        labels[last_index + 1: window_middle] = 3
                    last_index = window_middle

            row_counter = row_counter + 1
            pbar.update(1)

        pbar.close()
        labels = labels[1:last_index+1]
        return labels, last_index+1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Creation')
    parser.add_argument('--train', nargs="?", type=bool, default=False, help='training or testing')
    args = parser.parse_args()
    creator = DataCreator(128)
    if args.train:
        try:
            print("Training")
            print("Deleting old training data!")
            os.remove(current_folder + '/sp100_stocks.pkl')
            os.remove(current_folder + '/sp100_signals.pkl')
        except:
            print("Data retrieval started!")
        indices = training_indices
        creator.create_data(indices, stock_path=current_folder + '/sp100_stocks.pkl', label_path=current_folder + '/sp100_signals.pkl')
    else:
        print("Testing")
        try:
            print("Deleting old testing data!")
            os.remove(current_folder + '/test_stocks.pkl')
            os.remove(current_folder + '/test_signals.pkl')
        except:
            print("Data retrieval started!")
        indices = testing_indices
        creator.create_data(indices, stock_path=current_folder + '/test_stocks.pkl',
                            label_path=current_folder + '/test_signals.pkl')