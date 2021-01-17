import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def get_thin_params(layer, select_channels, dim=0):
    """
    Get params from layers after pruning
    """

    if isinstance(layer, nn.Conv2d):
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None

    elif isinstance(layer, nn.BatchNorm2d):
        assert dim == 0, "invalid dimension for bn_layer"

        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_mean = layer.running_mean.index_select(dim, select_channels)
        thin_var = layer.running_var.index_select(dim, select_channels)
        if layer.bias is not None:
            thin_bias = layer.bias.data.index_select(dim, select_channels)
        else:
            thin_bias = None
        return (thin_weight, thin_mean), (thin_bias, thin_var)
    elif isinstance(layer, nn.PReLU):
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_bias = None

    return thin_weight, thin_bias


def replace_layer(old_layer, init_weight, init_bias=None, keeping=False):
    """
    Replace specific layer of model
    :params layer: original layer
    :params init_weight: thin_weight
    :params init_bias: thin_bias
    :params keeping: whether to keep MaskConv2d
    """

    if hasattr(old_layer, "bias") and old_layer.bias is not None:
        bias_flag = True
    else:
        bias_flag = False

    if isinstance(old_layer, nn.Conv2d):
        if old_layer.groups != 1:
            new_groups = init_weight.size(0)
            in_channels = init_weight.size(0)
            out_channels = init_weight.size(0)
        else:
            new_groups = 1
            in_channels = init_weight.size(1)
            out_channels = init_weight.size(0)

        new_layer = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=old_layer.kernel_size,
                              stride=old_layer.stride,
                              padding=old_layer.padding,
                              bias=bias_flag,
                              groups=new_groups)

        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, nn.BatchNorm2d):
        weight = init_weight[0]
        mean_ = init_weight[1]
        bias = init_bias[0]
        var_ = init_bias[1]
        new_layer = nn.BatchNorm2d(weight.size(0))
        new_layer.weight.data.copy_(weight)
        assert init_bias is not None, "batch normalization needs bias"
        new_layer.bias.data.copy_(bias)
        new_layer.running_mean.copy_(mean_)
        new_layer.running_var.copy_(var_)
    elif isinstance(old_layer, nn.PReLU):
        new_layer = nn.PReLU(init_weight.size(0))
        new_layer.weight.data.copy_(init_weight)
    elif isinstance(old_layer, nn.Linear):
        output_size, input_size = init_weight.size()
        new_layer = nn.Linear(input_size, output_size, bias=bias_flag)
        new_layer.weight.data.copy_(init_weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    else:
        assert False, "unsupport layer type:" + \
                      str(type(old_layer))
    return new_layer


def pruning_conv(module, last_select_idx=None, amount=0.5, ln=2, dim=0):
    # Conv + BN + ReLu6
    prune.ln_structured(module[0], name="weight", amount=amount, n=ln, dim=dim)
    mask = module[0].weight_mask
    select_idx = mask.view(mask.size(0), -1).sum(1).nonzero().squeeze()
    # replace conv
    thin_weight, thin_bias = get_thin_params(module[0], select_idx, dim=0)
    if last_select_idx is not None:
        module[0] = replace_layer(module[0], thin_weight[:, last_select_idx], thin_bias)
    else:
        module[0] = replace_layer(module[0], thin_weight, thin_bias)
    # replace bn
    thin_weight, thin_bias = get_thin_params(module[1], select_idx, dim=0)
    module[1] = replace_layer(module[1], thin_weight, thin_bias)
    return module, select_idx

def pruning_block5(module, amount=0.5, ln=2, dim=0):
    # Conv0(dw) + BN1 + ReLu2 + Conv3(pw) + BN4
    prune.ln_structured(module[0], name="weight", amount=amount, n=ln, dim=dim)
    mask = module[0].weight_mask
    select_idx = mask.view(mask.size(0), -1).sum(1).nonzero().squeeze()
    # replace conv0
    thin_weight, thin_bias = get_thin_params(module[0], select_idx, dim=0)
    module[0] = replace_layer(module[0], thin_weight, thin_bias)
    # replace bn1
    thin_weight, thin_bias = get_thin_params(module[1], select_idx, dim=0)
    module[1] = replace_layer(module[1], thin_weight, thin_bias)
    
    last_select_idx = select_idx
    prune.ln_structured(module[3], name="weight", amount=amount, n=ln, dim=dim)
    mask = module[3].weight_mask
    select_idx = mask.view(mask.size(0), -1).sum(1).nonzero().squeeze()
    # replace conv3
    thin_weight, thin_bias = get_thin_params(module[3], select_idx, dim=0)
    module[3] = replace_layer(module[3], thin_weight[:, last_select_idx], thin_bias)
    # replace bn4
    thin_weight, thin_bias = get_thin_params(module[4], select_idx, dim=0)
    module[4] = replace_layer(module[4], thin_weight, thin_bias)
    return module, select_idx

def pruning_block8(module, last_select_idx, amount=0.5, ln=2, dim=0):
    # Conv0(pw) + BN1 + ReLu2 + Conv3(dw) + BN4 + ReLu5 + Conv6(pw) + BN7
    prune.ln_structured(module[0], name="weight", amount=amount, n=ln, dim=dim)
    mask = module[0].weight_mask
    select_idx = mask.view(mask.size(0), -1).sum(1).nonzero().squeeze()
    # replace conv0
    thin_weight, thin_bias = get_thin_params(module[0], select_idx, dim=0)
    module[0] = replace_layer(module[0], thin_weight[:, last_select_idx], thin_bias)
    # replace bn1
    thin_weight, thin_bias = get_thin_params(module[1], select_idx, dim=0)
    module[1] = replace_layer(module[1], thin_weight, thin_bias)
    
    prune.ln_structured(module[3], name="weight", amount=amount, n=ln, dim=dim)
    mask = module[3].weight_mask
    select_idx = mask.view(mask.size(0), -1).sum(1).nonzero().squeeze()
    # replace conv3
    thin_weight, thin_bias = get_thin_params(module[3], select_idx, dim=0)
    module[3] = replace_layer(module[3], thin_weight, thin_bias)
    # replace bn4
    thin_weight, thin_bias = get_thin_params(module[4], select_idx, dim=0)
    module[4] = replace_layer(module[4], thin_weight, thin_bias)

    last_select_idx = select_idx
    prune.ln_structured(module[6], name="weight", amount=amount, n=ln, dim=dim)
    mask = module[6].weight_mask
    select_idx = mask.view(mask.size(0), -1).sum(1).nonzero().squeeze()
    # replace conv6
    thin_weight, thin_bias = get_thin_params(module[6], select_idx, dim=0)
    module[6] = replace_layer(module[6], thin_weight[:, last_select_idx], thin_bias)
    # replace bn7
    thin_weight, thin_bias = get_thin_params(module[7], select_idx, dim=0)
    module[7] = replace_layer(module[7], thin_weight, thin_bias)
    return module, select_idx

def pruning_model(model, amount=0.5, ln=2, dim=0):
    # from mobilenetv2 import BaseMobileNetV2
    # model = BaseMobileNetV2()
    _, select_idx = pruning_conv(model.features[0], amount=amount, ln=ln, dim=dim)
    _, select_idx = pruning_block5(model.features[1].conv, amount=amount, ln=ln, dim=dim)
    for i in range(2, 18):
        _, select_idx = pruning_block8(model.features[i].conv, select_idx, amount=amount, ln=ln, dim=dim)
    _, select_idx = pruning_conv(model.features[18], select_idx, amount=amount, ln=ln, dim=dim)
    model.classifier[1] = replace_layer(model.classifier[1], 
                                        init_weight=model.classifier[1].weight[:, select_idx],
                                        init_bias=model.classifier[1].bias)
    return model