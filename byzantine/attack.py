import torch
import random


# not use Byzantine attack
def no_byz(v):
    return v


# replace the original gradient scalars by generated Gaussian noises
def gaussian_attack(v, num_attack):
    if num_attack == 0:
        return v
    else:
        indices = random.sample(range(len(v)), num_attack)
        for i in indices:
            for tensor in v[i]:
                tensor.copy_(torch.normal(0, 200, size=tensor.shape))
    return v


# the original gradient scalars <- the original gradient scalars * -1
def sign_attack(v, num_attack):
    if num_attack == 0:
        return v
    else:
        indices = random.sample(range(len(v)), num_attack)
        for i in indices:
            for tensor in v[i]:
                tensor.neg_()
    return v


def trim_attack(v, num_attack):
    # if num_attack == 0:
    #     return v
    # else:
    #     vi_shape = v[0].shape
    #     v_tran = torch.cat(v, dim=1)
    #     maximum_dim = torch.max(v_tran, dim=1)[0].reshape(vi_shape)
    #     minimum_dim = torch.min(v_tran, dim=1)[0].reshape(vi_shape)
    #     direction = torch.sign(torch.sum(torch.cat(v, dim=1), dim=-1, keepdim=True))
    #     directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

    #     for i in range(num_attack):
    #         random_12 = 1 + torch.rand(vi_shape)
    #         v[i] = directed_dim * (
    #             (direction * directed_dim > 0) / random_12
    #             + (direction * directed_dim < 0) * random_12
    #         )
    return v


def krum_attack(v, f):
    # if f == 0:
    #     return v
    # else:
    return v
