import torch
import torch.nn as nn
from torch.optim import Adam
from data_creation import DataCreator
from networks import Net
import os
import matplotlib.pyplot as plt
current_folder = os.getcwd()


class Model:
    def __init__(self, path=current_folder, learning_rate=0.01):
        self.path = path
        self.data_creator = DataCreator()
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
        self.criterion = nn.MSELoss()
        self.optimiser = Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, patience=30, min_lr=1e-9)

    def train(self, epochs):

        losses = []
        data_loader = self.data_creator.provide_training_stock()

        # train the network
        for epoch in range(epochs):

            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().unsqueeze(1).cuda()

                self.net.zero_grad()
                output = self.net(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimiser.step()

                # Print some loss stats
                if i % 200 == 0:
                    # append losses
                    losses.append((loss.item()))
                    # print  losses
                    print('Epoch [{:5d}/{:5d}] | loss: {:6.4f}'.format(
                        epoch + 1, epochs, loss.item()))
        print("Training completed!")
        plt.plot(losses)
        plt.show()

    def save(self):
        torch.save(self.net, self.path + "/net.pth")


if __name__ == "__main__":
    model = Model()
    model.train(10)
    model.save()