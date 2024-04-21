import torch
from torch import nn
from torch.nn.utils import prune
from typing import Optional


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
            module_weight_sparsity[name + ".weight"] = layer_sparsity_rate

    return module_weight_sparsity


def generate_random_mask_arr(
    model: nn.Module, sparsity_dict: dict[str, float], device: Optional[str] = None
) -> torch.Tensor:
    ret = []
    for name, p in model.named_parameters():
        n_elem_p = p.numel()

        if name in sparsity_dict:
            sparsity = sparsity_dict[name]
            # make sure at least 1 non_zero_count
            non_zero_count = max(int(torch.tensor(n_elem_p * (1 - sparsity))), 1)

            non_zero_index = torch.randperm(n_elem_p, device=device)[:non_zero_count]
            p_mask = torch.zeros((n_elem_p), device=device, dtype=bool)
            p_mask[non_zero_index] = True
        else:
            p_mask = torch.ones((n_elem_p,), device=device, dtype=bool)

        ret += [p_mask]

    return torch.concat(ret)
