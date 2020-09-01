import torch
import torch.nn as nn
import torch.optim as optim
from data_creation import DataCreator
from networks import Net
from tqdm import tqdm
from ralamb import Ralamb
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
        try:
            self.net = torch.load(self.path + "/net.pth")
            print("--------------------------------\n"
                  "Models were loaded successfully! \n"
                  "--------------------------------")
        except:
            print("-----------------------\n"
                  "No models were loaded! \n"
                  "-----------------------")
            self.net = Net(input_dim=225, hidden_dim=450)
        self.net.cuda()

    def predict_signal(self, ticker):
        signals = ['SELL', 'BUY', 'HOLD', 'STOP']
        _, data = get_daily_data(ticker, compact=True)
        self.net.train(False)
        with torch.no_grad():
            input = torch.tensor(data.to_numpy()[-1]).float().cuda()
            output = F.softmax(self.net(input), dim=-1).cpu().numpy()
            signal_idx = np.argmax(output)
        return signals[int(signal_idx)], 100*output[signal_idx]

    def test(self):

        losses = []
        accuracies = []
        buy_accuracies = []
        sell_accuracies = []
        up_accuracies = []
        down_accuracies = []
        data_loader = self.data_creator.provide_testing_stock()
        criterion = nn.CrossEntropyLoss()
        self.net.train(False)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.long().cuda()

                output = self.net(batch_x)
                loss = criterion(output, batch_y)

                output_metric = np.argmax(F.softmax(output, dim=1).cpu().numpy(), axis=1)
                batch_size = batch_y.size()[0]
                batch_y = batch_y.cpu().numpy()
                sell_mask_label = batch_y == 0
                sell_mask_output = output_metric == 0
                sell_accuracies.append(100*(sell_mask_label == sell_mask_output).sum()/batch_size)
                buy_mask_label = batch_y == 1
                buy_mask_output = output_metric == 1
                buy_accuracies.append(100*(buy_mask_label == buy_mask_output).sum()/batch_size)
                up_mask_label = batch_y == 2
                up_mask_output = output_metric == 2
                up_accuracies.append(100*(up_mask_label == up_mask_output).sum()/batch_size)
                down_mask_label = batch_y == 3
                down_mask_output = output_metric == 3
                down_accuracies.append(100 * (down_mask_label == down_mask_output).sum() / batch_size)
                losses.append((loss.item()))
                accuracy = 100 * sum(1 if output_metric[k] == batch_y[k] else 0 for k in
                                     range(batch_size)) / batch_size
                accuracies.append(accuracy)
        print("Average loss: ", np.mean(losses))
        print("Average accuracy: ", np.mean(accuracies))
        print("Buy-Average accuracy: ", np.mean(buy_accuracies))
        print("Sell-Average accuracy: ", np.mean(sell_accuracies))
        print("Hold-Average accuracy: ", np.mean(up_accuracies))
        print("Stop-Average accuracy: ", np.mean(down_accuracies))

    def display_annotated_graphs(self, ticker):
        imperatives = ['SELL', 'BUY', 'HOLD', 'STOP']
        stock, data = get_daily_data(ticker, True)
        with torch.no_grad():
            data = torch.from_numpy(scale_data(data)).float().cuda()
            signals = self.net(data)
            signals = np.argmax(F.softmax(signals, dim=1).cpu().numpy(), axis=1)
        stock = stock['close']
        plt.plot(stock.to_numpy())
        num_signals = signals.size
        for l in range(num_signals):
            plt.annotate(xy=(l, stock[l]), text=imperatives[signals[l]])
        plt.show()

    def train(self, epochs):

        rocs_aucs = []
        baseline_rocs_aucs = []
        losses = []
        accuracies = []
        data_loader, class_weights = self.data_creator.provide_training_stock()
        criterion = nn.CrossEntropyLoss()
        optimiser = optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-5, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=220, min_lr=1e-9)
        self.net.train(True)
        pbar = tqdm(total=epochs)

        # train the network
        for epoch in range(epochs):

            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.long().cuda()

                self.net.zero_grad()
                output = self.net(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimiser.step()

                scheduler.step(loss.item())

                # Print some loss stats
                if i % 2 == 0:
                    batch_size = batch_y.size()[0]
                    output_metric = F.softmax(output.detach().cpu(), dim=1).numpy()
                    random_metric = relabel_data(np.random.choice([0, 1, 2, 3], size=(1, batch_size), p=[1/4, 1/4, 1/4, 1/4]))
                    label_metric = relabel_data(batch_y.detach().cpu().numpy())
                    losses.append((loss.item()))
                    rocs_aucs.append(roc_auc_score(label_metric, output_metric, multi_class='ovo'))
                    baseline_rocs_aucs.append(roc_auc_score(label_metric, random_metric, multi_class='ovo'))
                    accuracy = 100 * sum(1 if np.argmax(output_metric[k]) == np.argmax(label_metric[k]) else 0 for k in
                                         range(batch_size)) / batch_size
                    accuracies.append(accuracy)
            pbar.update(1)
        pbar.close()
        fig, axs = plt.subplots(1, 3)
        axs[0].plot(np.convolve(losses, (1/25)*np.ones(25), mode='valid'))
        axs[1].plot(np.convolve(rocs_aucs, (1/25)*np.ones(25), mode='valid'))
        axs[1].plot(np.convolve(baseline_rocs_aucs, (1/25)*np.ones(25), mode='valid'))
        axs[1].legend(['Net', 'Baseline'])
        axs[2].plot(np.convolve(accuracies, (1/25)*np.ones(25), mode='valid'))
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
        model.train(1)
        model.save()
        print("Training completed!")
    else:
        model = Model()
        model.test()
        model.display_annotated_graphs('LYFT')
        print("Testing completed!")