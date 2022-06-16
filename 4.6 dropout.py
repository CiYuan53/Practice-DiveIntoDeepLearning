import torch
from torch import nn

from common import load_data_from_mnist, init_weights, train_ch3

if __name__ == "__main__":
    train_iter, test_iter = load_data_from_mnist()
    # add dropout layers within two hidden layers
    net = nn.Sequential(nn.Flatten(), 
        nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 10))
    net.apply(init_weights)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), 0.5)

    train_ch3(net, loss, optimizer, 5, train_iter, test_iter)
