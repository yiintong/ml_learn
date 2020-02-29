import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import util_pytorch as d2l
from collections import OrderedDict

print(torch.__version__)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


class LinearNet(nn.Module):
        def __init__(self, num_inputs, num_outputs):
                super(LinearNet, self).__init__()
                self.linear = nn.Linear(num_inputs, num_outputs)

        def forward(self, x):  # x shape: (batch, 1, 28, 28)
                y = self.linear(x.view(x.shape[0], -1))
                return y


net = LinearNet(num_inputs, num_outputs)


class FlattenLayer(nn.Module):
        def __init__(self):
                super(FlattenLayer, self).__init__()

        def forward(self, x):  # x shape: (batch, *, *, ...)
                return x.view(x.shape[0], -1)


net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))])
)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
# net = nn.Sequential(
#         d2l.FlattenLayer(),
#         nn.Linear(num_inputs, num_hiddens),
#         nn.ReLU(),
#         nn.Linear(num_hiddens, num_outputs),
#         )
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='E:\py_cache\dataset')
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)