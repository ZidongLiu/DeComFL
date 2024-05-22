"""
Model Parameter pruning using ZOO Grasp
Reference:
[1] Chen, Aochuan, et al. "Deepzero: Scaling up zeroth-order optimization for deep model training." arXiv preprint arXiv:2310.02025 (2023).
"""
import torch
from torch import nn
from torch.nn.utils import prune
from functools import partial
from typing import Union

from shared.model_helpers import eval_network_and_get_loss
from gradient_estimators.random_gradient_estimator import functional_forward_rge


def _fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx : idx + 1], targets[idx : idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break
    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat(
        [torch.cat(_) for _ in labels]
    ).view(-1)
    del dataloader_iter
    return X, y


def _extract_conv2d_and_linear_weights(model):

    if prune.is_pruned(model):
        return {
            f"{name}.weight_orig": m.weight_orig
            for name, m in model.named_modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        }
    else:
        return {
            f"{name}.weight": m.weight
            for name, m in model.named_modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        }


def _zoo_grasp_importance_score(
    model,
    dataloader,
    samples_per_class,
    class_num,
    num_pert: int,
    mu,
    loss_func=torch.nn.CrossEntropyLoss(),
):

    score_dict = {}
    device = next(model.parameters()).device
    x, y = _fetch_data(dataloader, class_num, samples_per_class)
    x, y = x.to(device), y.to(device)

    # only prune weight from conv2d and linear layer
    prune_params = _extract_conv2d_and_linear_weights(model)

    f_theta = partial(
        eval_network_and_get_loss, network=model, x=x, y=y, loss_func=loss_func
    )

    g0 = functional_forward_rge(f_theta, prune_params, num_pert, mu)
    modified_params = {}
    for key, param in prune_params.items():
        modified_params[key] = param.data + g0[key].data * mu
    g1 = functional_forward_rge(f_theta, modified_params, num_pert, mu)
    Hg = {}
    for key, param in prune_params.items():
        Hg[key] = (g1[key].data - g0[key].data) / mu

    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            # this line is copied from DeepZero, sometimes model already is pruned
            # thus we need to check weight_orig
            # in our use case, the truthy block should never be reached
            if hasattr(m, "weight_orig"):
                score_dict[(m, "weight")] = (
                    -m.weight_orig.clone().detach() * Hg[f"{name}.weight_orig"]
                )
            else:
                score_dict[(m, "weight")] = (
                    -m.weight.clone().detach() * Hg[f"{name}.weight"]
                )

    return score_dict


def zoo_grasp_prune(
    model: nn.Module,
    ratio: Union[float, torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    sample_per_classes=25,
    class_num: int = 10,
    num_pert: int = 1,
    mu: float = 1e-4,
):

    # NOTE: prune globally using score
    # the layer-wise pruning ratio will be used for layer-wise random pruning
    score_dict = _zoo_grasp_importance_score(
        model, dataloader, sample_per_classes, class_num, num_pert, mu
    )

    prune.global_unstructured(
        parameters=score_dict.keys(),
        pruning_method=prune.L1Unstructured,
        amount=ratio,
        importance_scores=score_dict,
    )


def layer_wise_random_prune(model: nn.Module, layer_wise_sparsity: dict[str, float]):
    for name, module in model.named_modules():
        if name in layer_wise_sparsity.keys():
            prune.random_unstructured(module, "weight", layer_wise_sparsity[name])
