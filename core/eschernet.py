import math
import json
from collections import deque
import torch.nn as nn

resnet = '[{"coords":[500,100],"layer_type":"CN","inputs":[],"outputs":[1],"layerConfig":{"in_channels":"3","out_channels":"64","kernel_size":"7","stride":"2","padding":"3"}},{"coords":[500,150],"layer_type":"BN","inputs":[0],"outputs":[2],"layerConfig":{"num_features":"64","epsilon":"0.00001","momentum":"0.1"}},{"coords":[500,200],"layer_type":"AC","inputs":[1],"outputs":[3],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[500,250],"layer_type":"PL","inputs":[2],"outputs":[4,9],"layerConfig":{"pool_type":"maxpool","kernel_size":"3","stride":"2","padding":"1"}},{"coords":[300,300],"layer_type":"CN","inputs":[3],"outputs":[5],"layerConfig":{"in_channels":"64","out_channels":"64","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[300,350],"layer_type":"BN","inputs":[4],"outputs":[6],"layerConfig":{"num_features":"64","epsilon":"0.00001","momentum":"0.1"}},{"coords":[300,400],"layer_type":"AC","inputs":[5],"outputs":[7],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[300,450],"layer_type":"CN","inputs":[6],"outputs":[8],"layerConfig":{"in_channels":"64","out_channels":"64","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[300,500],"layer_type":"BN","inputs":[7],"outputs":[9],"layerConfig":{"num_features":"64","epsilon":"0.00001","momentum":"0.1"}},{"coords":[300,550],"layer_type":"AC","inputs":[3,8],"outputs":[10,15],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[500,550],"layer_type":"CN","inputs":[9],"outputs":[11],"layerConfig":{"in_channels":"64","out_channels":"64","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[500,600],"layer_type":"BN","inputs":[10],"outputs":[12],"layerConfig":{"num_features":"64","epsilon":"0.00001","momentum":"0.1"}},{"coords":[500,650],"layer_type":"AC","inputs":[11],"outputs":[13],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[500,700],"layer_type":"CN","inputs":[12],"outputs":[14],"layerConfig":{"in_channels":"64","out_channels":"64","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[500,750],"layer_type":"BN","inputs":[13],"outputs":[15],"layerConfig":{"num_features":"64","epsilon":"0.00001","momentum":"0.1"}},{"coords":[500,800],"layer_type":"AC","inputs":[9,14],"outputs":[16,54],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[300,850],"layer_type":"CN","inputs":[15],"outputs":[17],"layerConfig":{"in_channels":"64","out_channels":"128","kernel_size":"3","stride":"2","padding":"1"}},{"coords":[300,900],"layer_type":"BN","inputs":[16],"outputs":[18],"layerConfig":{"num_features":"128","epsilon":"0.00001","momentum":"0.1"}},{"coords":[300,950],"layer_type":"AC","inputs":[17],"outputs":[19],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[300,1000],"layer_type":"CN","inputs":[18],"outputs":[20],"layerConfig":{"in_channels":"128","out_channels":"128","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[300,1050],"layer_type":"BN","inputs":[19],"outputs":[21],"layerConfig":{"num_features":"128","epsilon":"0.00001","momentum":"0.1"}},{"coords":[300,1100],"layer_type":"AC","inputs":[20,55],"outputs":[22,27],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[500,1150],"layer_type":"CN","inputs":[21],"outputs":[23],"layerConfig":{"in_channels":"128","out_channels":"128","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[500,1200],"layer_type":"BN","inputs":[22],"outputs":[24],"layerConfig":{"num_features":"128","epsilon":"0.00001","momentum":"0.1"}},{"coords":[500,1250],"layer_type":"AC","inputs":[23],"outputs":[25],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[500,1300],"layer_type":"CN","inputs":[24],"outputs":[26],"layerConfig":{"in_channels":"128","out_channels":"128","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[500,1350],"layer_type":"BN","inputs":[25],"outputs":[27],"layerConfig":{"num_features":"128","epsilon":"0.00001","momentum":"0.1"}},{"coords":[500,1400],"layer_type":"AC","inputs":[21,26],"outputs":[28,56],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[300,1450],"layer_type":"CN","inputs":[27],"outputs":[29],"layerConfig":{"in_channels":"128","out_channels":"256","kernel_size":"3","stride":"2","padding":"1"}},{"coords":[300,1500],"layer_type":"BN","inputs":[28],"outputs":[30],"layerConfig":{"num_features":"256","epsilon":"0.00001","momentum":"0.1"}},{"coords":[300,1550],"layer_type":"AC","inputs":[29],"outputs":[31],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[300,1600],"layer_type":"CN","inputs":[30],"outputs":[32],"layerConfig":{"in_channels":"256","out_channels":"256","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[300,1650],"layer_type":"BN","inputs":[31],"outputs":[33],"layerConfig":{"num_features":"256","epsilon":"0.00001","momentum":"0.1"}},{"coords":[300,1700],"layer_type":"AC","inputs":[32,57],"outputs":[34,39],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[500,1750],"layer_type":"CN","inputs":[33],"outputs":[35],"layerConfig":{"in_channels":"256","out_channels":"256","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[500,1800],"layer_type":"BN","inputs":[34],"outputs":[36],"layerConfig":{"num_features":"256","epsilon":"0.00001","momentum":"0.1"}},{"coords":[500,1850],"layer_type":"AC","inputs":[35],"outputs":[37],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[500,1900],"layer_type":"CN","inputs":[36],"outputs":[38],"layerConfig":{"in_channels":"256","out_channels":"256","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[500,1950],"layer_type":"BN","inputs":[37],"outputs":[39],"layerConfig":{"num_features":"256","epsilon":"0.00001","momentum":"0.1"}},{"coords":[500,2000],"layer_type":"AC","inputs":[33,38],"outputs":[40,58],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[300,2050],"layer_type":"CN","inputs":[39],"outputs":[41],"layerConfig":{"in_channels":"256","out_channels":"512","kernel_size":"3","stride":"2","padding":"1"}},{"coords":[300,2100],"layer_type":"BN","inputs":[40],"outputs":[42],"layerConfig":{"num_features":"512","epsilon":"0.00001","momentum":"0.1"}},{"coords":[300,2150],"layer_type":"AC","inputs":[41],"outputs":[43],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[300,2200],"layer_type":"CN","inputs":[42],"outputs":[44],"layerConfig":{"in_channels":"512","out_channels":"512","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[300,2250],"layer_type":"BN","inputs":[43],"outputs":[45],"layerConfig":{"num_features":"512","epsilon":"0.00001","momentum":"0.1"}},{"coords":[300,2300],"layer_type":"AC","inputs":[44,59],"outputs":[46,53],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[500,2350],"layer_type":"CN","inputs":[45],"outputs":[47],"layerConfig":{"in_channels":"512","out_channels":"512","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[500,2400],"layer_type":"BN","inputs":[46],"outputs":[48],"layerConfig":{"num_features":"512","epsilon":"0.00001","momentum":"0.1"}},{"coords":[500,2450],"layer_type":"AC","inputs":[47],"outputs":[49],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[500,2500],"layer_type":"CN","inputs":[48],"outputs":[50],"layerConfig":{"in_channels":"512","out_channels":"512","kernel_size":"3","stride":"1","padding":"1"}},{"coords":[500,2550],"layer_type":"BN","inputs":[49],"outputs":[53],"layerConfig":{"num_features":"512","epsilon":"0.00001","momentum":"0.1"}},{"coords":[300,2650],"layer_type":"PL","inputs":[53],"outputs":[52],"layerConfig":{"pool_type":"avgpool","kernel_size":"7","stride":"7","padding":"0"}},{"coords":[300,2700],"layer_type":"AF","inputs":[51],"outputs":[],"layerConfig":{"in_features":"512","out_features":"1000"}},{"coords":[300,2600],"layer_type":"AC","inputs":[45,50],"outputs":[51],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[550,900],"layer_type":"CN","inputs":[15],"outputs":[55],"layerConfig":{"in_channels":"64","out_channels":"128","kernel_size":"1","stride":"2","padding":"0"}},{"coords":[550,950],"layer_type":"BN","inputs":[54],"outputs":[21],"layerConfig":{"num_features":"128","epsilon":"0.00001","momentum":"0.1"}},{"coords":[600,1500],"layer_type":"CN","inputs":[27],"outputs":[57],"layerConfig":{"in_channels":"128","out_channels":"256","kernel_size":"1","stride":"2","padding":"0"}},{"coords":[600,1550],"layer_type":"BN","inputs":[56],"outputs":[33],"layerConfig":{"num_features":"256","epsilon":"0.00001","momentum":"0.1"}},{"coords":[650,2100],"layer_type":"CN","inputs":[39],"outputs":[59],"layerConfig":{"in_channels":"256","out_channels":"512","kernel_size":"1","stride":"2","padding":"0"}},{"coords":[650,2150],"layer_type":"BN","inputs":[58],"outputs":[45],"layerConfig":{"num_features":"512","epsilon":"0.00001","momentum":"0.1"}}]'
mnist_net = '[{"coords":[50,200],"layer_type":"CN","inputs":[],"outputs":[1],"layerConfig":{"in_channels":"1","out_channels":"10","kernel_size":"5","stride":"1","padding":"0"}},{"coords":[50,250],"layer_type":"PL","inputs":[0],"outputs":[2],"layerConfig":{"pool_type":"maxpool","kernel_size":"2","stride":"","padding":"0"}},{"coords":[50,300],"layer_type":"AC","inputs":[1],"outputs":[3],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[300,200],"layer_type":"CN","inputs":[2],"outputs":[4],"layerConfig":{"in_channels":"10","out_channels":"20","kernel_size":"5","stride":"1","padding":"0"}},{"coords":[300,250],"layer_type":"DR","inputs":[3],"outputs":[5],"layerConfig":{"percent":"0.5"}},{"coords":[300,300],"layer_type":"PL","inputs":[4],"outputs":[6],"layerConfig":{"pool_type":"maxpool","kernel_size":"2","stride":"","padding":"0"}},{"coords":[300,350],"layer_type":"AC","inputs":[5],"outputs":[7],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[550,200],"layer_type":"RS","inputs":[6],"outputs":[8],"layerConfig":{"x":"-1","y":"320"}},{"coords":[550,250],"layer_type":"AF","inputs":[7],"outputs":[9],"layerConfig":{"in_features":"320","out_features":"50"}},{"coords":[550,300],"layer_type":"AC","inputs":[8],"outputs":[10],"layerConfig":{"activation_fn":"ReLU"}},{"coords":[550,350],"layer_type":"AF","inputs":[9],"outputs":[11],"layerConfig":{"in_features":"50","out_features":"10"}},{"coords":[550,400],"layer_type":"AC","inputs":[10],"outputs":[],"layerConfig":{"activation_fn":"log_softmax"}}]'


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
        reshape_layers = filter(lambda n: n[1]['layer_type'] == 'RS', enumerate(self.network))
        reshape_inds = list(map(lambda n: n[0], reshape_layers))
        # print(reshape_inds)
        for idx, module in enumerate(self.network_modules):
            if idx not in reshape_inds:
                self.add_module(str(idx), module)
        
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


