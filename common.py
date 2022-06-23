from tokenize import Number
import torch
from torch import Tensor, nn
from torch.utils import data
from torchvision import transforms, datasets
from typing import Iterator, Tuple


def synthetic_data(w: torch.Tensor, b: torch.Tensor, num_examples: int):
    X = torch.normal(0, 1, (num_examples, w.shape[0]))
    y_vec = torch.matmul(X, w) + b
    y_vec += torch.normal(0, 0.01, y_vec.shape)
    y: torch.Tensor = y_vec.reshape((-1, 1))
    return X, y


def load_data_arrays(
    data_arrays: Tuple[torch.Tensor, ...], batch_size: int, shuffle=True
):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle)


def load_data_from_mnist(batch_size=200, resize: int = None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = datasets.FashionMNIST(
        "/Users/sunny/Desktop/D2L/data.nosync", True, trans, download=True
    )
    mnist_test = datasets.FashionMNIST(
        "/Users/sunny/Desktop/D2L/data.nosync", False, trans, download=True
    )
    return (
        data.DataLoader(mnist_train, batch_size, True),
        data.DataLoader(mnist_test, batch_size, False),
    )


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def count_correct(y_hat: torch.Tensor, y: torch.Tensor):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(1)  # convert into index of the largest element on dim 1
    cmp = y_hat.type(y.dtype) == y  # convert into the same type to avoid weird bug
    return cmp.type(y.dtype).sum().item()


def train_ch3(
    net: nn.Sequential,
    train_iter: Iterator,
    test_iter: Iterator,
    num_epochs: int,
    learning_rate: float,
):
    def init_weights(m: nn.Module):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.01)

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), learning_rate)

    net.train()
    for epoch in range(num_epochs):
        metric = Accumulator(3)  # loss and accuracy respected to the *train* dataset
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l: torch.Tensor = loss(y_hat, y)
            l.backward()  # *need* to do mean() if reduction was set to 'none'
            optimizer.step()
            metric.add(l.item(), count_correct(y_hat, y), y.numel())

        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        print(
            f"epoch {epoch + 1}: loss {train_loss:.8f}, accuracy {100 * train_acc:.2f}%"
        )

    metric = Accumulator(2)  # loss and accuracy respected to the *test* dataset
    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            metric.add(count_correct(net(X), y), y.numel())

    test_acc = metric[0] / metric[1]
    print(f"\ntest accuracy: {100 * test_acc:.2f}%")


def corr2d(X: torch.Tensor, K: torch.Tensor):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()

    return Y


def train_ch6(  # no gpu yet
    net: nn.Sequential,
    train_iter: Iterator,
    test_iter: Iterator,
    num_epochs: int,
    learning_rate: float,
):
    def init_weights(m: nn.Module):
        type_module = type(m)
        if type_module == nn.Linear or type_module == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), learning_rate)

    net.train()
    for epoch in range(num_epochs):
        metric = Accumulator(3)  # loss and accuracy respected to the *train* dataset
        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l: torch.Tensor = loss(y_hat, y)
            l.backward()  # *need* to do mean() if reduction was set to 'none'
            optimizer.step()
            metric.add(l.item(), count_correct(y_hat, y), y.numel())

        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        print(
            f"epoch {epoch + 1}: loss {train_loss:.8f}, accuracy {100 * train_acc:.2f}%"
        )

    metric = Accumulator(2)  # loss and accuracy respected to the *test* dataset
    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            metric.add(count_correct(net(X), y), y.numel())

    test_acc = metric[0] / metric[1]
    print(f"\ntest accuracy: {100 * test_acc:.2f}%")
