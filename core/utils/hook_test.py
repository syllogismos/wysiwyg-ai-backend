from core.eschernet import EscherNet
from core.mongo_queries import getNNModelById

from torch.autograd import Variable
import torch
import torch.nn as nn
from functools import reduce
import operator

no_of_elems = lambda t: reduce(operator.mul, list(t.size()), 1)


# nnm = getNNModelById('5a2986c0d53e4dc3af967cfe')
# model = EscherNet(nnm['network'])
# input = Variable(torch.randn(1, 1, 28, 28))

resnnm = getNNModelById('5a2986b5d53e4dc3af967cfd')
model = EscherNet(resnnm['network'])
input = Variable(torch.randn(1, 3, 300, 300))


target = Variable(torch.LongTensor([3]))
loss_fn = nn.CrossEntropyLoss()

def printnorm(self, input, output):
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # # input is a tuple of packed inputs
    # # output is a Variable. output.data is the Tensor we are interested
    # print('Inside ' + self.__class__.__name__ + ' forward')
    # print('')
    # print('input: ', type(input))
    # print('input[0]: ', type(input[0]))
    # print('output: ', type(output))
    # print('')
    # print('input size:', input[0].size())
    # print('output size:', output.data.size())
    # print('output norm:', output.data.norm())
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    pass

def forward_hook(self, input, output):
    self.register_buffer('input_buffer', input)
    self.register_buffer('output_buffer', output)
    

def printgradnorm(self, grad_input, grad_output):
    # print('************************************************')
    # print(self)
    # print('Inside ' + self.__class__.__name__ + ' backward')
    # print('Inside class:' + self.__class__.__name__)
    # print('')
    # print('grad_input: ', type(grad_input))
    # print('grad_input[0]: ', type(grad_input[0]))
    # print('grad_output: ', type(grad_output))
    # print('grad_output[0]: ', type(grad_output[0]))
    # print('')
    # # print('grad_input size:', grad_input[0].size())
    # print('grad_output size:', grad_output[0].size())
    # print('grad_input norm:', grad_output[0].data.norm())
    # print('************************************************')
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
    

# h1 = model.network_modules[0].register_forward_hook(printnorm)
# hb1 = model.network_modules[0].register_backward_hook(printgradnorm)
h = []
hb = []
for module in model.modules():
    # print(module)
    # hb.append(module.register_backward_hook(printgradnorm))
    h.append(module.register_forward_hook(forward_hook))


out = model(input)
err = loss_fn(out, target)
err.backward()
"""
[(m, getattr(model, str(m)).es_grad_input) for m in list(range(12)) if m not in model.reshape_inds]
[(m, getattr(model, str(m)).es_grad_output[0].size()) for m in list(range(12)) if m not in model.reshape_inds]
[(m, getattr(model, str(m)).input_buffer) for m in list(range(12)) if m not in model.reshape_inds]
[(m, getattr(model, str(m)).output_buffer.size()) for m in list(range(12)) if m not in model.reshape_inds]
"""
print([getattr(model, str(m)).output_norm for m in list(range(12)) if m not in model.reshape_inds])
out = model(input)
err = loss_fn(out, target)
err.backward()
print([getattr(model, str(m)).output_norm for m in list(range(12)) if m not in model.reshape_inds])
x = list(model.named_modules())
y = list(model.named_parameters())



