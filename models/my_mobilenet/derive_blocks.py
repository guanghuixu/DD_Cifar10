import torch
import torch.nn as nn

# from ..tools.utils import parse_net_config, sort_net_config
from .operations import OPS, conv_bn

def parse_net_config(net_config):
    if isinstance(net_config, list):
        return net_config
    elif isinstance(net_config, str):
        str_configs = net_config.split('|')
        return [eval(str_config) for str_config in str_configs]
    else:
        raise TypeError

def load_net_config(path):
    with open(path, 'r') as f:
        net_config = ''
        while True:
            line = f.readline().strip()
            if line:
                net_config += line
            else:
                break
    return net_config

def sort_net_config(net_config):
    def sort_skip(op_list):
        # put the skip operation to the end of one stage
        num = op_list.count('skip')
        for _ in range(num):
            op_list.remove('skip')
            op_list.append('skip')
        return op_list

    sorted_net_config = []
    for cfg in net_config:
        cfg[1] = sort_skip(cfg[1])
        sorted_net_config.append(cfg)
    return sorted_net_config


class Block(nn.Module):
    def __init__(self, in_ch, block_ch, ops, stride, use_se=False, bn_params=[True, True]):
        super(Block, self).__init__()
        layers = []

        for i, op in enumerate(ops):
            if i == 0:
                block_stride = stride
                block_in_ch = in_ch
            else:
                block_stride = 1
                block_in_ch = block_ch
            block_out_ch = block_ch

            layers.append(OPS[op](block_in_ch, block_out_ch, block_stride, 1, 
                                        affine=bn_params[0], track=bn_params[1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def derive_blocks(net_config, if_sort=True):
    parsed_net_config = parse_net_config(net_config)
    if if_sort:
        parsed_net_config = sort_net_config(parsed_net_config)
    blocks = nn.ModuleList()
    blocks.append(conv_bn(3, parsed_net_config[0][0][0], 2))
    for cfg in parsed_net_config:
        blocks.append(Block(cfg[0][0], cfg[0][1], cfg[1], cfg[2]))
    return blocks, parsed_net_config
