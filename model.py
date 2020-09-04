import torch
import torch.nn as nn
import torch.optim as optim
from data_creation import DataCreator
from networks import SimpleNet, DenseNet3
from tqdm import tqdm
import optimizers as opt
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from stats import relabel_data, get_daily_data
import torch.nn.functional as F
from stats import scale_data

current_folder = os.getcwd()


class Model:
    def __init__(self, path=current_folder, learning_rate=1e-3, batch_size=128):
        torch.manual_seed(12345)
        self.path = path
        self.data_creator = DataCreator(batch_size)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        try:
            self.net = torch.load(self.path + "/net.pth")
            print("--------------------------------\n"
                  "Models were loaded successfully! \n"
                  "--------------------------------")
        except:
            print("-----------------------\n"
                  "No models were loaded! \n"
                  "-----------------------")
            self.net = SimpleNet(225, 450) # DenseNet3(depth=40, num_classes=4, bottleneck=True, growth_rate=36, drop_rate=0.2)
        self.net.cuda()

    def predict_signal(self, ticker):
        signals = ['SELL', 'BUY', 'HOLD', 'STOP']
        _, data = get_daily_data(ticker, compact=True)
        self.net.train(False)
        with torch.no_grad():
            input = torch.tensor(data.to_numpy()[-1]).reshape(15, 15).unsqueeze(0).unsqueeze(0).float().cuda()
            output = self.net(input).cpu().numpy().flatten()
            signal_idx = np.argmax(output)
        return signals[int(signal_idx)], 100 * output[signal_idx]

    def test(self):

        losses = []
        accuracies = []
        buy_accuracies = []
        sell_accuracies = []
        up_accuracies = []
        down_accuracies = []
        data_loader = self.data_creator.provide_testing_stock()
        criterion = nn.MSELoss()
        self.net.train(False)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.unsqueeze(1).float().cuda()
                batch_y = batch_y.long().cuda()

                output = self.net(batch_x)
                loss = criterion(output, batch_y)
                accuracy, buy_accuracy, sell_accuracy, up_accuracy, down_accuracy = self.log_performance(output,
                                                                                                         batch_y)
                losses.append((loss.item()))
                accuracies.append(accuracy)
                buy_accuracies.append(buy_accuracy)
                sell_accuracies.append(sell_accuracy)
                down_accuracies.append(down_accuracy)
                up_accuracies.append(up_accuracy)

        print("Average loss: ", np.mean(losses))
        print("Average accuracy: ", np.mean(accuracies))
        print("Buy-Average accuracy: ", np.mean(buy_accuracies))
        print("Sell-Average accuracy: ", np.mean(sell_accuracies))
        print("Hold-Average accuracy: ", np.mean(up_accuracies))
        print("Stop-Average accuracy: ", np.mean(down_accuracies))

    def log_performance(self, output, batch_y):
        output_metric = np.argmax(output.detach().cpu().numpy(), axis=1)
        batch_y = np.argmax(batch_y.detach().cpu().numpy(), axis=1)
        batch_size = batch_y.size
        sell_mask_label = batch_y == 0
        sell_mask_output = output_metric == 0
        sell_accuracy = 100 * (sell_mask_label == sell_mask_output).sum() / batch_size
        buy_mask_label = batch_y == 1
        buy_mask_output = output_metric == 1
        buy_accuracy = 100 * (buy_mask_label == buy_mask_output).sum() / batch_size
        up_mask_label = batch_y == 2
        up_mask_output = output_metric == 2
        up_accuracy = 100 * (up_mask_label == up_mask_output).sum() / batch_size
        down_mask_label = batch_y == 3
        down_mask_output = output_metric == 3
        down_accuracy = 100 * (down_mask_label == down_mask_output).sum() / batch_size
        accuracy = 100 * sum(1 if output_metric[k] == batch_y[k] else 0 for k in
                             range(batch_size)) / batch_size
        return accuracy, buy_accuracy, sell_accuracy, up_accuracy, down_accuracy

    def display_annotated_graphs(self, ticker):
        imperatives = ['SELL', 'BUY', 'HOLD', 'STOP']
        stock, data = get_daily_data(ticker, True)
        with torch.no_grad():
            data = torch.from_numpy(scale_data(data)).reshape(-1, 15, 15).unsqueeze(1).float().cuda()
            signals = self.net(data)
            signals = np.argmax(F.softmax(signals, dim=1).cpu().numpy(), axis=1)
        stock = stock['close']
        plt.plot(stock.to_numpy())
        num_signals = signals.size
        for l in range(num_signals):
            plt.annotate(xy=(l, stock[l]), text=imperatives[signals[l]])
        plt.show()

    def train(self):

        losses = []
        accuracies = []
        buy_accuracies = []
        sell_accuracies = []
        up_accuracies = []
        down_accuracies = []
        data_loader, n = self.data_creator.provide_training_stock()
        criterion = nn.MSELoss()
        optimiser = opt.LookaheadAdam(self.net.parameters(), lr=self.learning_rate, weight_decay=5e-5, alpha=1.0, k=3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.7, patience=100, min_lr=1e-10)
        self.net.train(True)
        pbar = tqdm(range(int(n / self.batch_size)))

        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.unsqueeze(1).float().cuda()
            batch_y = batch_y.float().cuda()
            self.net.zero_grad()
            output = self.net(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimiser.step()
            scheduler.step(loss.item())
            if i % 2 == 0:
                accuracy, buy_accuracy, sell_accuracy, up_accuracy, down_accuracy = self.log_performance(output,
                                                                                                         batch_y)
                losses.append((loss.item()))
                accuracies.append(accuracy)
                buy_accuracies.append(buy_accuracy)
                sell_accuracies.append(sell_accuracy)
                down_accuracies.append(down_accuracy)
                up_accuracies.append(up_accuracy)
            pbar.update(1)
        pbar.close()

        print("Average loss: ", np.mean(losses[-100:]))
        print("Average accuracy: ", np.mean(accuracies[-100:]))
        print("Buy-Average accuracy: ", np.mean(buy_accuracies[-100:]))
        print("Sell-Average accuracy: ", np.mean(sell_accuracies[-100:]))
        print("Hold-Average accuracy: ", np.mean(up_accuracies[-100:]))
        print("Stop-Average accuracy: ", np.mean(down_accuracies[-100:]))
        self.plot_performance_log(losses, buy_accuracies, sell_accuracies, up_accuracies, down_accuracies, accuracies)

    def plot_performance_log(self, losses, buy_accuracies, sell_accuracies, up_accuracies, down_accuracies, accuracies):
        fig, axs = plt.subplots(1, 3)
        axs[0].plot(np.convolve(losses, (1 / 25) * np.ones(25), mode='valid'))
        axs[1].plot(np.convolve(buy_accuracies, (1 / 25) * np.ones(25), mode='valid'))
        axs[1].plot(np.convolve(sell_accuracies, (1 / 25) * np.ones(25), mode='valid'))
        axs[1].plot(np.convolve(up_accuracies, (1 / 25) * np.ones(25), mode='valid'))
        axs[1].plot(np.convolve(down_accuracies, (1 / 25) * np.ones(25), mode='valid'))
        axs[1].legend(['BUY', 'SELL', 'HOLD', 'STOP'])
        axs[2].plot(np.convolve(accuracies, (1 / 25) * np.ones(25), mode='valid'))
        plt.show()

    def save(self):
        torch.save(self.net, self.path + "/net.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--train', nargs="?", type=bool, default=False, help='training or testing')
    args = parser.parse_args()
    if args.train:
        try:
            print("Deleting old net!")
            os.remove(current_folder + '/net.pth')
        except:
            print("Training started!")
        model = Model()
        model.train()
        model.save()
        print("Training completed!")
    else:
        model = Model()
        model.test()
        model.display_annotated_graphs('LYFT')
        print("Testing completed!")
