from __future__ import print_function
import torch.nn as nn
import math
import json
from collections import deque

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from core.eschernet import EscherNet, mnist_net
# from utils import create_timed_rotating_log
args = {
    'cuda': False,
    'seed': 1,
    'batch_size': 64,
    'test_batch_size': 1000,
    'epochs': 2,
    'lr': 0.01,
    'momentum': 0.5,
    'log_interval': 10
}
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# log = create_timed_rotating_log('./train.log', 'mnsit')

torch.manual_seed(args['seed'])
if args['cuda']:
    torch.cuda.manual_seed(args['seed'])


kwargs = {'num_workers': 1, 'pin_memory': True} if args['cuda'] else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['test_batch_size'], shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# model = Net()



def mnist_train(model, optimizer, epoch, log):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            log.info('mnist',
                exp = '353f9c5',
                train_log = {
                    'epoch': epoch,
                    'loss': loss.data[0],
                    'batch_idx': batch_idx
                }
                # epoch = epoch,
                # loss = loss.data[0],
                # batch_idx = batch_idx
            )
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.data[0]))

def mnist_test(model, log):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))



def experiment(log):
    model = EscherNet(mnist_net)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    for epoch in range(1, args['epochs'] + 1):
        mnist_train(model, optimizer, epoch, log)
        mnist_test(model, log)
