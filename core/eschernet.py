import math
import json
from collections import deque
import torch.nn as nn

mnist_net = '[{"coords":[50,200],"layer_type":"CN","inputs":[],"outputs":[1],"layerConfig":{"in_channels":"1","stride":"1","out_channels":"10","kernel_size":"5","layer_id":0,"padding":"0"}},{"coords":[50,250],"layer_type":"PL","inputs":[0],"outputs":[2],"layerConfig":{"padding":"0","pool_type":"maxpool","stride":"","kernel_size":"2","layer_id":1}},{"coords":[50,300],"layer_type":"AC","inputs":[1],"outputs":[3],"layerConfig":{"activation_fn":"ReLU","layer_id":2}},{"coords":[300,200],"layer_type":"CN","inputs":[2],"outputs":[4],"layerConfig":{"in_channels":"10","stride":"1","out_channels":"20","kernel_size":"5","layer_id":3,"padding":"0"}},{"coords":[300,250],"layer_type":"DR","inputs":[3],"outputs":[5],"layerConfig":{"percent":"0.5","layer_id":4}},{"coords":[300,300],"layer_type":"PL","inputs":[4],"outputs":[6],"layerConfig":{"padding":"0","pool_type":"maxpool","stride":"","kernel_size":"2","layer_id":5}},{"coords":[300,350],"layer_type":"AC","inputs":[5],"outputs":[7],"layerConfig":{"activation_fn":"ReLU","layer_id":6}},{"coords":[550,200],"layer_type":"RS","inputs":[6],"outputs":[8],"layerConfig":{"y":"320","x":"-1","layer_id":7}},{"coords":[550,250],"layer_type":"AF","inputs":[7],"outputs":[9],"layerConfig":{"in_features":"320","out_features":"50","layer_id":8}},{"coords":[550,300],"layer_type":"AC","inputs":[8],"outputs":[10],"layerConfig":{"activation_fn":"ReLU","layer_id":9}},{"coords":[550,350],"layer_type":"AF","inputs":[9],"outputs":[11],"layerConfig":{"in_features":"50","out_features":"10","layer_id":10}},{"coords":[550,400],"layer_type":"AC","inputs":[10],"outputs":[],"layerConfig":{"activation_fn":"log_softmax","layer_id":11}}]'

class EscherNet(nn.Module):

    def __init__(self, fabricNetwork):
        """
        Parses the json obtained from the dashboard
        and computes feedforward on that network.abs

        Check if the network passes basic checks, like is it dag,
        or have a single input and single output and etc.
        """
        super(EscherNet, self).__init__()
        self.network = json.loads(fabricNetwork)
        self.network_modules = list(map(parse_single_node, self.network))
        self.network_layer_ids = list(map(lambda x: x['layerConfig']['layer_id'], self.network))
        # all layers must have unique layer id, we use this layer id as the id of the module in this module
        assert len(self.network_layer_ids) == len(set(self.network_layer_ids))
        reshape_layers = filter(lambda n: n[1]['layer_type'] == 'RS', enumerate(self.network))
        self.reshape_inds = list(map(lambda n: n[0], reshape_layers))
        # print(reshape_inds)
        self.backward_hook_registers = []
        for idx, module in enumerate(self.network_modules):
            if idx not in self.reshape_inds:
                self.backward_hook_registers.append(
                    module.register_backward_hook(save_grads_to_buffer_hook)
                )
                self.add_module(str(self.network_layer_ids[idx]), module)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.weight.data.fill_(0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.fill_(0.5)
            #     m.bias.data.zero_()

        self.topological_sort = get_topological_sort(self.network)
        self.parent = self.topological_sort[0]
        self.last_node = self.topological_sort[-1]

    def forward(self, x):
        for i in self.topological_sort:
            if len(self.network[i]['inputs']) == 0:
                self.network[i]['x'] = self.network_modules[i](x)
            else:
                self.network[i]['x'] = self.network_modules[i](sum(map(lambda y: self.network[y]['x'], self.network[i]['inputs'])))
        # make this more general the final edge case, hardcoded reshape layer here to make resnet18 work
        # x = self.network[self.topological_sort[-2]]['x']
        # x = x.view(x.size(0), -1)
        # x = self.network_modules[self.last_node](x)
        # return x #self.network[self.last_node]['x']
        return self.network[self.last_node]['x']


def save_grads_to_buffer_hook(self, grad_input, grad_output):
    self.register_buffer('es_grad_output', grad_output)
    self.register_buffer('es_grad_input', grad_input)


def parse_single_node(node):
    """
    given a single node from escher network computes the respective
    nn function
    """
    layer_type = node['layer_type']
    config = node['layerConfig']
    if layer_type == 'CN':
        in_channels = int(config['in_channels'])
        out_channels = int(config['out_channels'])
        kernel_size = int(config['kernel_size'])
        stride = int(config['stride'])
        padding = int(config['padding'])
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    if layer_type == 'AF':
        in_features = int(config['in_features'])
        out_features = int(config['out_features'])
        return nn.Linear(in_features, out_features)
    if layer_type == 'AC':
        activation = config['activation_fn']
        if activation == 'ReLU':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'log_softmax':
            return nn.LogSoftmax()
        else:
            raise NotImplementedError
    if layer_type == 'BN':
        num_features = int(config['num_features'])
        epsilon = float(config['epsilon'])
        momentum = float(config['momentum'])
        return nn.BatchNorm2d(num_features, epsilon, momentum)
    if layer_type == 'DR':
        percent = float(config['percent'])
        return nn.Dropout2d(p=percent)
    if layer_type == 'PL':
        pool_type = config['pool_type']
        kernel_size = int(config['kernel_size'])
        try:
            stride = int(config['stride'])
        except:
            stride = None
        # print(stride, config['stride'])
        padding = int(config['padding'])
        if pool_type == 'maxpool':
            # print(stride)
            # print(nn.MaxPool2d(kernel_size, stride=stride, padding=padding))
            return nn.MaxPool2d(kernel_size, stride, padding)
        if pool_type == 'avgpool':
            return nn.AvgPool2d(kernel_size, stride, padding)
    if layer_type == 'RS':
        x = int(config['x'])
        y = int(config['y'])
        if x == 0:
            return lambda z: z.view(z.size(0), y)
        return lambda z: z.view(x, y)
    else:
        raise NotImplementedError

def get_topological_sort(network):
    """
    L ← Empty list that will contain the sorted elements
    S ← Set of all nodes with no incoming edge
    while S is non-empty do
        remove a node n from S
        add n to tail of L
        for each node m with an edge e from n to m do
            remove edge e from the graph
            if m has no other incoming edges then
                insert m into S
    if graph has edges then
        return error (graph has at least one cycle)
    else 
        return L (a topologically sorted order)
    """

    incoming_edges = list(enumerate(map(lambda x: deque(x['inputs']), network)))
    # incoming_edges = [(0, deque), (1, deque)]
    outgoing_edges = list(enumerate(map(lambda x: deque(x['outputs']), network)))
    # outgoing_edges = [(0, deque), (2, deque)]
    topological_sort = []
    parent_nodes = deque(filter(lambda x: len(x[1]) == 0, incoming_edges))
    # parent_nodes = deque([(0, deque)])
    while len(parent_nodes) > 0:
        node = parent_nodes.pop() # node = (0, deque)
        topological_sort.append(node)
        children = outgoing_edges[node[0]] # children = (0, deque[1])
        while len(children[1]) > 0:
            dest_node_index = children[1].pop()
            incoming_edges[dest_node_index][1].remove(node[0])
            if len(incoming_edges[dest_node_index][1]) == 0:
                parent_nodes.append(incoming_edges[dest_node_index])
    return list(map(lambda x: x[0], topological_sort))

            
# if __name__ == '__main__':
#     en = EscherNet(resnet)


