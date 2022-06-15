import torch
from torch import nn

from common import load_data_from_mnist, init_weights, train_ch3

if __name__ == "__main__":
    train_iter, test_iter = load_data_from_mnist()

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))  # flatten image 28x28
    net.apply(init_weights)  # another way to initialize the model

    loss = nn.CrossEntropyLoss()  # cross entropy, default reduction is mean()

    optimizer = torch.optim.SGD(net.parameters(), 0.1)  # sgd

    train_ch3(net, loss, optimizer, 5, train_iter, test_iter)
