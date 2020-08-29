import torch
import torch.nn as nn
from torch.optim import Adam
from data_creation import DataCreator
from networks import Net
from tqdm import tqdm
from ralamb import Ralamb
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from stats import relabel_data
import torch.nn.functional as F
current_folder = os.getcwd()


class Model:
    def __init__(self, path=current_folder, learning_rate=1e-3, batch_size=128):
        self.path = path
        self.batch_size = batch_size
        self.data_creator = DataCreator(self.batch_size)
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
            self.net = Net(input_dim=136, hidden_dim=272)
        self.net.cuda()

    def test(self):
        pass

    def train(self, epochs):

        rocs_aucs = []
        baseline_rocs_aucs = []
        losses = []
        accuracies = []
        data_loader, class_weights = self.data_creator.provide_training_stock()
        criterion = nn.CrossEntropyLoss()
        optimiser = Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=10, min_lr=1e-9)
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
                    output_metric = F.softmax(output.detach().cpu(), dim=1).numpy()
                    random_metric = relabel_data(np.random.choice([0, 1, 2], size=(1, self.batch_size), p=[1/3, 1/3, 1/3]))
                    label_metric = relabel_data(batch_y.detach().cpu().numpy())
                    losses.append((loss.item()))
                    rocs_aucs.append(roc_auc_score(label_metric, output_metric, multi_class='ovo'))
                    baseline_rocs_aucs.append(roc_auc_score(label_metric, random_metric, multi_class='ovo'))
                    accuracy = 100 * sum(1 if np.argmax(output_metric[k]) == np.argmax(label_metric[k]) else 0 for k in
                                         range(self.batch_size)) / self.batch_size
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
    try:
        print("Deleting old net!")
        os.remove(current_folder + '/net.pth')
    except:
        print("Training started!")
    model = Model()
    model.train(1)
    model.save()
    print("Training completed!")