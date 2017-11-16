from __future__ import print_function

from core.mongo_queries import getNNModelById, getExperimentById, getUserById
from core.mongo_queries import getDatasetById
from core.config import RLLAB_AMI

from core.eschernet import EscherNet
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import deque
import json, math, structlog
import boto3


ec2 = boto3.resource('ec2')




def launch_exp(exp):
    """
    given the exp object launch the relevant experiment

    exp: exp dict
    log: structlog instance with expid and userid
    """
    if exp['type'] == 'rl':
        launch_rl_exp(exp)
        pass
    elif exp['type'] == 'supervised':
        logger = structlog.get_logger('train_logs')
        log = logger.new(user=exp['user'], exp=str(exp['_id']))
        log.info('exp_timeline', timeline={'message': 'Experiment Launched', 'level': 'info'})
        launch_supervised_exp(exp, log)
        pass
    pass


def launch_rl_exp(exp):
    """
    exp: exp dict
    start spot/ec2 instances with half the prices as cutoff for each experiment variant
    """
    logger = structlog.get_logger('train_logs')
    log = logger.new(user=exp['user'], exp=str(exp['_id']))
    log.info('exp_timeline', timeline={'message': 'Experiment Launched'})
    no_of_variants = len(exp['config']['variants'])
    for variantIndex in range(no_of_variants):
        log.info('exp_timeline',  timeline={'message': 'Launching machine for variant %s' %variantIndex, 'level': 'info'})
        instance = ec2.create_instances(
            MaxCount=1,
            MinCount=1,
            ImageId=RLLAB_AMI,
            InstanceType=exp['config']['machine_type'],
            KeyName='facebook',
            NetworkInterfaces=[{
                'SubnetId': 'subnet-347a7e1c',
                'Groups': ['sg-8ed5a8eb'],
                'AssociatePublicIpAddress': True,
                'DeviceIndex': 0
            }],
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        { 'Key': 'Name', 'Value': 'RL Experiment' },
                        { 'Key': 'ExperimentId', 'Value': exp['_id'] },
                        { 'Key': 'VariantIndex', 'Value': variantIndex }
                    ]
                }
            ]
        )
        log.info('exp_timeline', timeline={
            'message': 'Machine successfully started for varint %s' %variantIndex,
            'variant': variantIndex,
            'instance_id': instance[0].instance_id,
            'private_ip': instance[0].private_ip_address,
            'level': 'debug'
        })

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

def launch_supervised_exp(exp, log):
    variants = exp['config']['variants']
    for idx, variant in enumerate(variants):
        supervised_exp_single_variant(exp, idx, log)

def supervised_exp_single_variant(exp, variant_idx, log):
    """
    launch a supervised experiment
    """
    # dataset = getDatasetById(exp['dataset'])
    dataset_name = exp['dataset']
    nnmodel = getNNModelById(exp['model'])
    network = nnmodel['network']

    exp_config = exp['config']

    # Creating model
    model = EscherNet(network)

    variant = exp['config']['variants'][variant_idx]

    # experiment settings
    lr = float(variant['lr'])
    momentum = float(variant['momentum'])
    epochs = int(variant['epochs'])
    batch_size = int(variant['batch_size'])
    test_batch_size = int(variant['test_batch_size'])
    seed = int(variant['seed'])

    # loss function
    loss_type = exp_config['loss']

    # Creating optimizer
    optimizer = optim.__dict__[exp_config['optim']]

    nn_optimizer = optimizer(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = F.__dict__[loss_type]

    # Loading dataset

    if dataset_name == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=True, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307, ), (0.3081, ))
                        ])),
                        batch_size=batch_size, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
            batch_size=test_batch_size, shuffle=True
        )
    elif dataset_name == 'tiny-imagenet-test':
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                './data/tiny-imagenet-200/train',
                transforms.Compose([
                    transforms.RandomSizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            ),
            batch_size=batch_size, shuffle=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                './data/tiny-imagenet-200/val',
                transforms.Compose([
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            ),
            batch_size=test_batch_size, shuffle=False
        )
    log_info = {
        'variant': variant_idx
    }
    # Traning
    for epoch in range(1, epochs + 1):
        supervised_train(train_loader, model, nn_optimizer, loss_fn, epoch, log, log_info)
        supervised_test(test_loader, model, loss_fn, epoch, log, log_info)


def supervised_train(train_loader, model, nn_optimizer, loss_fn, epoch, log, log_info):
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
                         'batch_idx': batch_idx,
                         'variant': log_info['variant']
                     }
            )

    pass


def supervised_test(test_loader, model, loss_fn, epoch, log, log_info):
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
    log.info('val_log',
             test_log = {
                 'epoch': epoch,
                 'loss': test_loss,
                 'variant': log_info['variant']
             }
    )
    pass








