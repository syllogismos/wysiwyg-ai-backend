from core.eschernet import EscherNet
from core.mongo_queries import getNNModelById
import torch
from torch.autograd import Variable
import numpy as np
from numpy import random


RESNET18 = "5a2986b5d53e4dc3af967cfd"
UNET = "5a8be4b9a42b57d6ae8043a7"

def reset_seed():
    random.seed(1)
    torch.manual_seed(1)


if __name__ == '__main__':
    reset_seed()
    x = Variable(torch.FloatTensor(np.random.random((1, 3, 320, 320))))
    unet_model = EscherNet(getNNModelById(UNET)['network'])
    out = unet_model(x)
    print(torch.sum(out), torch.sum(x))
    print("1e5*6.8318, 1e5*1.5350 above results should be these with both numpy and torch seed as 1")

    reset_seed()
    resnet_model = EscherNet(getNNModelById(RESNET18)['network'])
    out = resnet_model(x)
    # print(out.size())
    print(torch.sum(out), torch.sum(x))
    print("1e-2*9.4395, 1e5*1.5350 above results should be these with both numpy and torch seed as 1")
