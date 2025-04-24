import random

import torch


# not use Byzantine attack
def no_byz(v: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
    return v


# replace the original gradient scalars by generated Gaussian noises
def gaussian_attack(v: list[list[torch.Tensor]], num_attack: int) -> list[list[torch.Tensor]]:
    if num_attack == 0:
        return v
    else:
        indices = random.sample(range(len(v)), num_attack)
        for i in indices:
            for tensor in v[i]:
                tensor.copy_(torch.normal(0, 200, size=tensor.shape))
    return v


# the original gradient scalars <- the original gradient scalars * -1
def sign_attack(v: list[list[torch.Tensor]], num_attack: int) -> list[list[torch.Tensor]]:
    if num_attack == 0:
        return v
    else:
        indices = random.sample(range(len(v)), num_attack)
        for i in indices:
            for tensor in v[i]:
                tensor.neg_()
    return v


def trim_attack(v: list[list[torch.Tensor]], num_attack: int) -> list[list[torch.Tensor]]:
    num_pert = int(v[0][0].shape[0])
    vi_shape = num_pert * len(v[0])
    v_tran = torch.stack([torch.cat(one_client_data, dim=0) for one_client_data in v])
    max_values, _ = torch.max(v_tran, dim=0)
    min_values, _ = torch.min(v_tran, dim=0)
    direction = torch.sign(torch.sum(v_tran, dim=0))
    directed_dim = (direction > 0) * min_values + (direction < 0) * max_values
    for i in range(num_attack):
        random_12 = 1 + torch.rand(size=(vi_shape,)).to(v[0][0].device)
        p_v = directed_dim * (
            (direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12
        )
        v[i] = [p_v[s : s + num_pert] for s in range(0, vi_shape, num_pert)]
    return v


def score(gradient: torch.Tensor, v: torch.Tensor, f: int = 1) -> float:
    num_neighbours = int(v.shape[1] - 2 - f)
    sorted_distance, _ = torch.square(v - gradient).sum(dim=0).sort()
    return sorted_distance[1 : (1 + num_neighbours)].sum().item()


def krum(v: list[list[torch.Tensor]], f: int = 1):
    v_tran = torch.stack([torch.cat(one_client_data, dim=0) for one_client_data in v])
    scores = torch.tensor([score(gradient, v_tran, f) for gradient in v_tran])
    min_idx = int(torch.argmin(scores, dim=0).item())
    krum_nd = v[min_idx]
    return min_idx, krum_nd


def krum_attack(v: list[list[torch.Tensor]], f: int, lr: float) -> list[list[torch.Tensor]]:
    if f == 0:
        return v
    else:
        epsilon = 0.01
        num_pert = int(v[0][0].shape[0])
        vi_shape = num_pert * len(v[0])
        _, original_dir = krum(v, f)
        original_dir = torch.stack(original_dir)
        lamda = 0.25
        for i in range(f):
            temp = -lamda * torch.sign(original_dir)
            split_tensors = torch.unbind(temp, dim=0)
            v[i] = list(split_tensors)
        min_idx, _ = krum(v, f)
        stop_threshold = 0.00001 * 2 / lr
        while min_idx >= f and lamda > stop_threshold:
            lamda = lamda / 2
            for i in range(f):
                temp = -lamda * torch.sign(original_dir)
                split_tensors = torch.unbind(temp, dim=0)
                v[i] = list(split_tensors)
            min_idx, _ = krum(v, f)
        v[0] = -lamda * torch.sign(original_dir)  # type: ignore[call-overload]
        for i in range(1, f):
            random_raw = torch.rand(vi_shape) - 0.5
            random_norm = torch.rand(1).item() * epsilon / lr
            randomness = random_raw * random_norm / torch.norm(random_raw)
            v[i] = -lamda * torch.sign(original_dir) + randomness.to(v[0][0].device)
    return v
