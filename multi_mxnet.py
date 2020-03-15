import torch
import numpy as np
import sys
import utils as d2l
batch_size = 256
import torchvision.datasets as dsets
import  torchvision.transforms as transforms
from  PIL import Image
DOWNLOAD = False
transform1 = transforms.ToTensor()  #可以把下载到的数据转化成张量格式
#transforms.Compose()定义多重数据变化
transform2 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  #归一化[-1,1]

# todo for yxt
# have bug  ,result from mxnet framwork
# it may be have a dataset problem
# mT_trainset = dsets.MNIST(root="E:\py_cache\dataset",train=True,transform=transform1,download=DOWNLOAD)
# mT_testset = dsets.MNIST(root="E:\py_cache\dataset",train=False,transform=transform1,download=DOWNLOAD)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,root='E:\py_cache\dataset')

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

loss = torch.nn.CrossEntropyLoss()

num_epochs, lr = 5, 100.0
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

print('nihao')