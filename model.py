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
current_folder = os.getcwd()


class Model:
    def __init__(self, path=current_folder, learning_rate=1e-4, batch_size=128):
        self.path = path
        self.batch_size = batch_size
        self.data_creator = DataCreator(self.batch_size)
        try:
            self.net = torch.load(self.path + "/net.pth")
            print("--------------------------------\n"
                  "Models were loaded successfully! \n"
                  "--------------------------------")
        except:
            print("-----------------------\n"
                  "No models were loaded! \n"
                  "-----------------------")
            self.net = Net(input_dim=96, hidden_dim=192)
        self.net.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = Ralamb(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, patience=30, min_lr=1e-9)

    def train(self, epochs):

        accuracies = []
        losses = []
        data_loader = self.data_creator.provide_training_stock()
        self.net.train(True)
        pbar = tqdm(total=epochs)

        # train the network
        for epoch in range(epochs):

            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.long().cuda()

                self.net.zero_grad()
                output = self.net(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimiser.step()

                # Print some loss stats
                if i % 10 == 0:
                    output_metric = np.argmax(output.detach().cpu().numpy(), axis=1).flatten()
                    label_metric = batch_y.detach().cpu().numpy().flatten()
                    accuracy = 100*sum(1 if output_metric[k] == label_metric[k] else 0 for k in range(self.batch_size))/self.batch_size
                    losses.append((loss.item()))
                    accuracies.append(accuracy)
            pbar.update(1)
        pbar.close()
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(losses)
        axs[1].plot(accuracies)
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
    model.train(2)
    model.save()
    print("Training completed!")