import torch
from torch import nn

from common import synthetic_data, load_data_arrays

if __name__ == "__main__":
    real_w = torch.ones(200) * 0.01
    real_b = torch.tensor(0.05)
    train_iter = load_data_arrays(synthetic_data(real_w, real_b, 20), 5)
    test_iter = load_data_arrays(synthetic_data(real_w, real_b, 100), 5, False)

    net = nn.Sequential(nn.Linear(200, 1))

    loss = nn.MSELoss()

    wd = 0  # 0 means no weight decay

    optimizer = torch.optim.SGD(net.parameters(), 0.003, weight_decay=wd)

    for epoch in range(10):
        for X, y in train_iter:
            optimizer.zero_grad()
            loss(net(X), y).backward()
            optimizer.step()
    print("w's L2 norm: ", net[0].weight.norm().item())  # converts one-element tensor to scalar
