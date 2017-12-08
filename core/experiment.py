from __future__ import print_function

from core.mongo_queries import getNNModelById, getExperimentById, getUserById
from core.mongo_queries import getDatasetById
from core.config import RLLAB_AMI, RL_USER_DATA, HOME_DIR
from core.config import SUPERVISED_AMI, SUP_USER_DATA

from core.eschernet import EscherNet
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import deque
import json, math, structlog, operator, os
import boto3
from functools import reduce
# from core.utils.bad_grad_viz import register_hooks


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
        launch_sup_exp(exp)
        # logger = structlog.get_logger('train_logs')
        # log = logger.new(user=exp['user'], exp=str(exp['_id']))
        # log.info('exp_timeline', timeline={'message': 'Experiment Launched', 'level': 'info'})
        # launch_supervised_exp(exp, log)
        pass
    pass

def launch_sup_exp(exp):
    """
    exp: exp dict
    start ec2 instances
    """
    logger = structlog.get_logger('train_logs')
    log = logger.new(user=exp['user'], exp=str(exp['_id']))
    log.info('exp_timeline', timeline={'message': 'Experiment Launched', 'level': 'info'})
    no_of_variants = len(exp['config']['variants'])
    print(no_of_variants)
    for variantIndex in range(no_of_variants):
        instance = ec2.create_instances(
            MaxCount=1,
            MinCount=1,
            ImageId=SUPERVISED_AMI,
            InstanceType=exp['config']['machine_type'],
            UserData=SUP_USER_DATA,
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
                        { 'Key': 'Name', 'Value': 'Sup Experiment' },
                        { 'Key': 'ExperimentId', 'Value': str(exp['_id']) },
                        { 'Key': 'VariantIndex', 'Value': str(variantIndex) }
                    ]
                }
            ]
        )
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(instance)
        log.info('exp_timeline', timeline={
            'message': 'Machine successfully started for variant %s' %variantIndex,
            'variant': variantIndex,
            'instance_id': instance[0].instance_id,
            'private_ip': instance[0].private_ip_address,
            'level': 'info'
        })

def launch_rl_exp(exp):
    """
    exp: exp dict
    start spot/ec2 instances with half the prices as cutoff for each experiment variant
    """
    logger = structlog.get_logger('train_logs')
    log = logger.new(user=exp['user'], exp=str(exp['_id']))
    log.info('exp_timeline', timeline={'message': 'Experiment Launched', 'level': 'info'})
    no_of_variants = len(exp['config']['variants'])
    print(no_of_variants)
    for variantIndex in range(no_of_variants):
        # log.info('exp_timeline',  timeline={'message': 'Launching machine for variant %s' %variantIndex, 'level': 'info'})
        instance = ec2.create_instances(
            MaxCount=1,
            MinCount=1,
            ImageId=RLLAB_AMI,
            InstanceType=exp['config']['machine_type'],
            UserData=RL_USER_DATA,
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
                        { 'Key': 'ExperimentId', 'Value': str(exp['_id']) },
                        { 'Key': 'VariantIndex', 'Value': str(variantIndex) }
                    ]
                }
            ]
        )
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(instance)
        log.info('exp_timeline', timeline={
            'message': 'Machine successfully started for variant %s' %variantIndex,
            'variant': variantIndex,
            'instance_id': instance[0].instance_id,
            'private_ip': instance[0].private_ip_address,
            'level': 'info'
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
            datasets.MNIST(os.path.join(HOME_DIR, 'data/mnist'), train=True, download=True, 
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
                os.path.join(HOME_DIR, 'data/tiny-imagenet-200/train'),
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
                os.path.join(HOME_DIR, 'data/tiny-imagenet-200/val'),
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
        checkpoint_epoch(model, nn_optimizer, epoch, log_info, exp, log)
        supervised_test(test_loader, model, loss_fn, epoch, log, log_info)

def checkpoint_epoch(model, nn_optimizer, epoch, log_info, exp, log):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': nn_optimizer.state_dict(),
        'experiment': exp['_id'],
        'variant': log_info['variant']
    }
    filename = os.path.join(HOME_DIR, 'results', 'checkpoint_%s.pth'%epoch)
    torch.save(checkpoint, filename)

    layer_stats = {}
    for layer_id in model.network_layer_ids:
        if layer_id[1] != 'RS':
            layer_stats[layer_id[0]] = get_layer_stats(model,layer_id[0])
    stats_filename = os.path.join(HOME_DIR, 'results/stats_%s.json'%epoch)
    json.dump(layer_stats, open(stats_filename, 'w'))

def get_layer_stats(model, layer_id):
    no_of_elems = lambda t: reduce(operator.mul, list(t.size()), 1) # no of elements in a tensor, multiply all the dimensions
    module = getattr(model, str(layer_id))
    grad_output_data = module.es_grad_output[0].data
    total = no_of_elems(grad_output_data)
    zeroes = grad_output_data.eq(0.0).sum()
    positive = grad_output_data.gt(0.0).sum()
    negative = grad_output_data.lt(0.0).sum()
    exploded = grad_output_data.ne(grad_output_data).any() or grad_output_data.gt(1e6).any()
    stats = {
        'norm': grad_output_data.norm(),
        'zeros': zeroes/total,
        'positive': positive/total,
        'negative': negative/total,
        'exploded': exploded
    }
    return stats


"""
    self.register_buffer('output_norm', grad_output[0].data.norm())
    self.register_buffer('es_grad_output', grad_output)
    self.register_buffer('es_grad_input', grad_input)
    total = no_of_elems(grad_output[0].data)
    zeroes = grad_output[0].data.eq(0.0).sum()
    positive = grad_output[0].data.gt(0.0).sum()
    negative = grad_output[0].data.lt(0.0).sum()
    self.register_buffer('grad_output_percent_zero', zeroes/total)
    self.register_buffer('grad_output_percent_pos', positive/total)
    self.register_buffer('grad_output_percent_neg', negative/total)
    self.register_buffer('grad_output_explode', grad_output[0].data.ne(grad_output[0].data).any() or grad_output[0].data.gt(1e6).any())
    
"""

def supervised_train(train_loader, model, nn_optimizer, loss_fn, epoch, log, log_info):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        nn_optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        # if batch_idx % 10 == 0:
        #     get_dot = register_hooks(loss)
        loss.backward()
        nn_optimizer.step()
        if batch_idx % 10 == 0:
            # dot = get_dot()
            # dot.save('dot_%s_%s.dot'%(epoch, batch_idx))
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
             val_log = {
                 'epoch': epoch,
                 'loss': test_loss,
                 'variant': log_info['variant']
             }
    )
    pass








