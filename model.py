import torch
from data_creation import DataCreator
from networks import Net


class Model:
    def __init__(self):
        self.data_creator = DataCreator()
        self.net = Net(input_dim=9, hidden_dim=100)
        self.net.cuda()

    def train(self, epochs):

        print_every = 50
        losses = []
        data_loader = self.data_creator.provide_training_stock()

        # train the network
        for epoch in range(epochs):

            for i, (batch_x, batch_y) in enumerate(data_loader):

                # Get batch size
                batch_size = batch_x.size()[0]

                # Scale the data
                train_x = torch.tensor(train_x)

                # print(train_x[:10])

                train_y = train_y.long()

                model.train()

                model.zero_grad()

                # In GPU
                if train_on_gpu:
                    train_x = train_x.float().cuda()
                    train_y = train_y.cuda()

                output = model(train_x)

                loss = criterion(output, train_y)

                loss.backward()
                optimizer.step()

                # Print some loss stats
                if batch_i % print_every == 0:
                    # append losses
                    losses.append((loss.item()))
                    # print  losses
                    print('Epoch [{:5d}/{:5d}] | loss: {:6.4f}'.format(
                        epoch + 1, epochs, loss.item()))

        return model