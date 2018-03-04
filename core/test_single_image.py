from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from core.eschernet import EscherNet
from core.mongo_queries import getNNModelById

CHECKPOINT_FILE = "chkp/checkpoint_27.pth"
IMAGE_NAME = "/home/ubuntu/data/val_simple/dog/dog.98.jpg"
CAT_IMAGE = "/home/ubuntu/data/val_simple/cat/cat.9806.jpg"
DOG_IMAGE = "/home/ubuntu/data/val_simple/dog/dog.98.jpg"
VALIDATION_FOLDER = "/home/ubuntu/data/val_simple/"
NETWORK_ID = "5a39e1f65086367e35a36158"


checkpoint = torch.load(CHECKPOINT_FILE)

for key in list(checkpoint['model'].keys()):
    if 'es_grad_output' in key or 'es_grad_input' in key:
        del(checkpoint['model'][key])

network = getNNModelById(NETWORK_ID)['network']

model = EscherNet(network)

loader = transforms.Compose([transforms.Scale(256), transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.autograd.Variable(image)
    image = image.unsqueeze(0)
    return image

model.load_state_dict(checkpoint['model'])
model.eval()

def test_single_image():
    print("test single image")
    cat_output = model(image_loader(CAT_IMAGE))
    dog_output = model(image_loader(DOG_IMAGE))


    criterion = nn.CrossEntropyLoss()
    _, cat_pred = cat_output.topk(2, 1, True, True)
    _, dog_pred = dog_output.topk(2, 1, True, True)
    print(cat_pred, dog_pred)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        VALIDATION_FOLDER,
        transforms.Compose([transforms.Scale(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])
                        ),
    batch_size=4, shuffle=False)


def test_validation():
    print("inside validation")
    for i, (input, target) in enumerate(val_loader):
        print(i)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        loss = nn.CrossEntropyLoss(output, target_var)
        #print(loss)
        topk = (2,)
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.data.topk(maxk, 1, True, True)
        print(pred)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))
        print(res)


if __name__ == '__main__':
    # test_single_image()
    test_validation()