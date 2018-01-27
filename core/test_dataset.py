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

CAT_MODEL_ID = "5a39e1f65086367e35a36158"


nnmodel = getNNModelById(CAT_MODEL_ID)
model = EscherNet(nnmodel['network'])

lr = 0.01
momentum = 0.5
epochs = 2
batch_size = 2
test_batch_size = 2
seed = 1
loss_fn = F.__dict__['nll_loss']
optimizer = optim.__dict__['SGD']
nn_optimizer = optimizer(model.parameters(), lr=lr, momentum=momentum)

train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(HOME_DIR, 'catdog/catdog/train'),
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
                os.path.join(HOME_DIR, 'catdog/catdog/val'),
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

print("Training one epoch starts")
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = Variable(data), Variable(target)
    nn_optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    print(batch_idx, loss.data[0])