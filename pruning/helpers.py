import torch
from torch.nn.utils import prune


def get_module_weight_sparsity(model):
    sum_list_all = 0
    zero_sum_all = 0
    module_weight_sparsity = {}

    for name, m in model.named_modules():

        if prune.is_pruned(m) and hasattr(m, "weight"):
            sum_list_all = sum_list_all + float(m.weight.nelement())
            zero_sum_all = zero_sum_all + float(torch.sum(m.weight == 0))

            sum_list = float(m.weight.nelement())
            zero_sum = float(torch.sum(m.weight == 0))

            layer_sparsity_rate = zero_sum / sum_list
            module_weight_sparsity[name] = layer_sparsity_rate

    return module_weight_sparsity
