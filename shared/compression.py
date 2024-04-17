import torch


def top_k(tensor, k: int):
    values, indices = torch.topk(tensor.view(-1), k)
    sparse_tensor = torch.zeros_like(tensor)
    sparse_tensor.view(-1)[indices] = values
    return sparse_tensor
