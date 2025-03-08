import torch
from torch import nn as nn


def module_has_string(has_strings, x):
    # No string required, all layers can be converted
    if len(has_strings) == 0:
        return True

    # Only modules with names contain one string in has_strings can be converted
    for string in has_strings:
        if string in x:
            return True
    return False


def has_string(has_strings, x):
    # No string required, all layers can be converted
    if len(has_strings) == 0:
        return False

    # Only modules with names contain one string in has_strings can be converted
    for string in has_strings:
        if string in x:
            return True
    return False


def collect_module_params(model, module_class, has_strings=[]):
    params = []
    names = []
    nnn = []
    for nm, m in model.named_modules():
        if has_string(nnn, nm):
            continue
        if isinstance(m, module_class) and module_has_string(has_strings, nm):
            for np, p in m.named_parameters():
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def set_param_trainable(model, module_names, requires_grad):
    classes = {
        'bn': nn.BatchNorm2d,
        'conv': nn.Conv2d,
        'fc': nn.Linear
    }
    set_names = []
    for name in module_names:
        params, param_names = collect_module_params(model, classes[name])
        for p in params:
            p.requires_grad = requires_grad
        set_names.extend(param_names)
    return set_names


def remove_param_grad(model, module_names):
    classes = {
        'bn': nn.BatchNorm2d,
        'conv': nn.Conv2d,
        'fc': nn.Linear
    }
    set_names = []
    for name in module_names:
        params, param_names = collect_module_params(model, classes[name])
        for p in params:
            p.grad = None
        set_names.extend(param_names)
    return set_names

# 这里momentum的值就是online?但是好像不止影响这块
def get_optimizer(opt_dic, lr, opt_type='sgd', momentum=True):
    if opt_type == 'sgd':
        opt = torch.optim.SGD(opt_dic, lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(opt_dic, lr=lr, betas=(0.9, 0.999))
    return opt


# def get_new_optimizers(model, lr=1e-4, names=('bn', 'conv', 'fc'), opt_type='sgd', momentum=False):
#     optimizers, opt_names = [], []
#     classes = {
#         'bn': nn.BatchNorm2d,
#         'conv': nn.Conv2d,
#         'fc': nn.Linear
#     }
    # opt_dic = []
    # for name in names:
    #     name = name.lower()
    #     params, names = collect_module_params(model, module_class=classes[name])
    #     for param in params:
    #         opt_dic.append({'params': param, 'lr': lr})
    # opt = get_optimizer(opt_dic, lr, opt_type, momentum)
    # # optimizers.append(opt)
    # # opt_names.append(names)
    # return opt


def get_new_optimizers(model, lr=1e-4, names=('bn', 'conv', 'fc'), opt_type='sgd', momentum=False):
    opt_dic = []
    classes = {
        'bn': nn.BatchNorm2d,
        'conv': nn.Conv2d,
        'fc': nn.Linear,
        'ln': nn.LayerNorm  # 如果你希望支持LayerNorm，也可以添加这一项
    }
    for name in names:
        name = name.lower()
        params_group, paras_name = collect_module_params(model, module_class=classes[name])
        if len(params_group)==0 and name=='bn':
            print('Try to convert to Transformer-based Layernorm...')
            name = 'ln'
            params_group, paras_name = collect_module_params(model, module_class=classes[name])

        if params_group:  # 如果当前组有参数则添加
            for param in params_group:
                opt_dic.append({'params': param, 'lr': lr})
    if not opt_dic:
        raise ValueError("No parameters found for the given module types: {}".format(names))
    opt = get_optimizer(opt_dic, lr, opt_type, momentum)
    return opt

"""
ori
"""
def convert_to_target(net, norm, start=0, end=5, verbose=True, res50=False):
    def convert_norm(old_norm, new_norm, num_features, idx):
        norm_layer = new_norm(num_features, idx=idx).to(net.conv1.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [0, net.layer1, net.layer2, net.layer3, net.layer4]

    idx = 0
    converted_bns = {}
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 0:
            net.bn1 = convert_norm(net.bn1, norm, net.bn1.num_features, idx)
            converted_bns['L0-BN0-0'] = net.bn1
            idx += 1
        else:
            for j, bottleneck in enumerate(layer):
                bottleneck.bn1 = convert_norm(bottleneck.bn1, norm, bottleneck.bn1.num_features, idx)
                converted_bns['L{}-BN{}-{}'.format(i, j, 0)] = bottleneck.bn1
                idx += 1
                bottleneck.bn2 = convert_norm(bottleneck.bn2, norm, bottleneck.bn2.num_features, idx)
                converted_bns['L{}-BN{}-{}'.format(i, j, 1)] = bottleneck.bn2
                idx += 1
                if res50:
                    bottleneck.bn3 = convert_norm(bottleneck.bn3, norm, bottleneck.bn3.num_features, idx)
                    converted_bns['L{}-BN{}-{}'.format(i, j, 3)] = bottleneck.bn3
                    idx += 1
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = convert_norm(bottleneck.downsample[1], norm, bottleneck.downsample[1].num_features, idx)
                    converted_bns['L{}-BN{}-{}'.format(i, j, 2)] = bottleneck.downsample[1]
                    idx += 1
    return net, converted_bns


"""
Vit
"""
# def convert_to_target(net, norm, start=0, end=5, verbose=True, res50=False): 
#     def convert_norm(old_norm, new_norm, num_features, idx):
#         # ViT 中 LayerNorm 的转换
#         norm_layer = new_norm(num_features).to(net.vit.conv_proj.weight.device)  # 修改为 net.vit.conv_proj
#         if hasattr(norm_layer, 'load_old_dict'):
#             info = 'Converted to : {}'.format(norm)
#             norm_layer.load_old_dict(old_norm)
#         elif hasattr(norm_layer, 'load_state_dict'):
#             state_dict = old_norm.state_dict()
#             info = norm_layer.load_state_dict(state_dict, strict=False)
#         else:
#             info = 'No load_old_dict() found!!!'
#         if verbose:
#             print(info)
#         return norm_layer

#     # ViT 中的层级结构，使用 net.vit.encoder.layers 访问
#     layers = net.vit.encoder.layers  # Sequential 中的所有层，包含多个 EncoderBlock

#     idx = 0
#     converted_layers = {}
    
#     # 遍历 layers，逐个访问 EncoderBlock
#     for i, layer in enumerate(layers):
#         if not (start <= i < end):
#             continue
#         # 对每个 EncoderBlock 中的 LayerNorm 进行转换
#         if hasattr(layer, 'ln_1') and isinstance(layer.ln_1, nn.LayerNorm):  # 确保是 LayerNorm 层
#             layer.ln_1 = convert_norm(layer.ln_1, norm, layer.ln_1.normalized_shape[0], idx)
#             converted_layers['L{}-LN{}-{}'.format(i, 0, 0)] = layer.ln_1
#             idx += 1
#         if hasattr(layer, 'ln_2') and isinstance(layer.ln_2, nn.LayerNorm):  # 确保是 LayerNorm 层
#             layer.ln_2 = convert_norm(layer.ln_2, norm, layer.ln_2.normalized_shape[0], idx)
#             converted_layers['L{}-LN{}-{}'.format(i, 0, 1)] = layer.ln_2
#             idx += 1

#         # 处理 MLP 中的 LayerNorm 层
#         for j, sub_layer in enumerate(layer.mlp):
#             if isinstance(sub_layer, nn.LayerNorm):  # 确保是 LayerNorm 层
#                 sub_layer = convert_norm(sub_layer, norm, sub_layer.normalized_shape[0], idx)
#                 converted_layers['L{}-MLP-LN{}-{}'.format(i, j, 0)] = sub_layer
#                 idx += 1

#     return net, converted_layers