import torch
from torch import nn

from common import synthetic_data, load_data_arrays


if __name__ == "__main__":
    real_w = torch.tensor([2, -3.4])
    real_b = torch.tensor(4.2)
    features, labels = synthetic_data(real_w, real_b, 100)
    data_iter = load_data_arrays((features, labels), 10)

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()  # mean squared error

    optimizer = torch.optim.SGD(net.parameters(), 0.03)

    for epoch in range(3):
        for X, y in data_iter:
            l: torch.Tensor = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        print(f"epoch {epoch + 1}: loss {loss(net(features), labels) :f}")
