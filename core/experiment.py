from __future__ import print_function

from core.mongo_queries import getNNModelById, getExperimentById, getUserById
from core.mongo_queries import getDatasetById

from core.eschernet import EscherNet
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import deque
import json, math




def launch_exp(exp, log):
    """
    given the exp object launch the relevant experiment

    exp: exp dict
    log: structlog instance with expid and userid
    """
    if exp['type'] == 'rl':
        pass
    elif exp['type'] == 'supervised':
        log.info('experiment launched')
        supervised_exp(exp, log)
        pass
    pass

dataset = {
    'user': '59f56c75f1c16a64a00eaaef',
    'name': 'MNIST',
}

nnmodel = {
    'user': '59f56c75f1c16a64a00eaaef',
    'name': 'MNIST',
    'network': '[{"coords":[50,200],"layer_type":"CN","inputs":[],"outputs":[1],"layerConfig":{"in_channels":"1","out_channels":"10","kernel_size":"5","stride":"1","padding":"0"}},{"coords":[50,250],"layer_type":"PL","inputs":[0],"outputs":[2],"layerConfig":{"pool_type":"maxpool","kernel_size":"2","stride":"","padding":"0"}},{"coords":[50,300],"layer_type":"AC","inputs":[1],"outputs":[3],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[300,200],"layer_type":"CN","inputs":[2],"outputs":[4],"layerConfig":{"in_channels":"10","out_channels":"20","kernel_size":"5","stride":"1","padding":"0"}},{"coords":[300,250],"layer_type":"DR","inputs":[3],"outputs":[5],"layerConfig":{"percent":"0.5"}},{"coords":[300,300],"layer_type":"PL","inputs":[4],"outputs":[6],"layerConfig":{"pool_type":"maxpool","kernel_size":"2","stride":"","padding":"0"}},{"coords":[300,350],"layer_type":"AC","inputs":[5],"outputs":[7],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[550,200],"layer_type":"RS","inputs":[6],"outputs":[8],"layerConfig":{"x":"-1","y":"320"}},{"coords":[550,250],"layer_type":"AF","inputs":[7],"outputs":[9],"layerConfig":{"in_features":"320","out_features":"50"}},{"coords":[550,300],"layer_type":"AC","inputs":[8],"outputs":[10],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[550,350],"layer_type":"AF","inputs":[9],"outputs":[11],"layerConfig":{"in_features":"50","out_features":"10"}},{"coords":[550,400],"layer_type":"AC","inputs":[10],"outputs":[],"layerConfig":{"activation_fn":"log_softmax"}}]'
}

# 59f595e68eb343059e2682c9
experiment = {
    'user': '59f56c75f1c16a64a00eaaef',
    'dataset': '59f5957a8eb343059e2682c7',
    'model': '59f595ad8eb343059e2682c8',
    'type': 'supervised',
    'name': 'MNIST',
    'config': {
        'lr': 0.01,
        'momentum': 0.5,
        'batch_size': 64,
        'test_batch_size': 1000,
        'seed': 1,
        'loss': 'nll_loss',
        'optim': 'SGD',
        'epochs': 5
    }
}

def supervised_exp(exp, log):
    """
    launch a supervised experiment
    """
    dataset = getDatasetById(exp['dataset'])
    nnmodel = getNNModelById(exp['model'])
    network = nnmodel['network']

    exp_config = exp['config']

    # Creating model
    model = EscherNet(network)

    # Creating optimizer
    optimizer = optim.__dict__[exp_config['optim']]

    # experiment settings
    lr = exp_config['lr']
    momentum = exp_config['momentum']
    epochs = exp_config['epochs']
    batch_size = exp_config['batch_size']
    test_batch_size = exp_config['test_batch_size']
    seed = exp_config['seed']
    loss = exp_config['loss']

    nn_optimizer = optimizer(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = F.__dict__[loss]

    # Loading dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307, ), (0.3081, ))
                       ])),
                       batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
        batch_size=test_batch_size, shuffle=True
    )

    # Traning
    for epoch in range(1, epochs + 1):
        supervised_train(train_loader, model, nn_optimizer, loss_fn, epoch, log)
        supervised_test(test_loader, model, loss_fn, epoch, log)


def supervised_train(train_loader, model, nn_optimizer, loss_fn, epoch, log):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        nn_optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        nn_optimizer.step()
        if batch_idx % 10 == 0:
            log.info('train_log',
                     train_log = {
                         'epoch': epoch,
                         'loss': loss.data[0],
                         'batch_idx': batch_idx
                     }
            )

    pass


def supervised_test(test_loader, model, loss_fn, epoch, log):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += loss_fn(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    log.info('test_log',
             test_log = {
                 'epoch': epoch,
                 'loss': test_loss
             }
    )
    pass








