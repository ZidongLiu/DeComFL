import torch


def top_k(tensor, k: int):
    values, indices = torch.topk(tensor.view(-1), k)
    sparse_tensor = torch.zeros_like(tensor)
    sparse_tensor.view(-1)[indices] = values
    return sparse_tensor


# To do: keep the random seed for two forward passes
def random_k(tensor, k: int):
    num_elements = tensor.numel()
    k = min(k, num_elements)
    non_zero_indices = torch.randperm(num_elements)[:k]
    sparse_tensor = torch.zeros_like(tensor)
    sparse_tensor.view(-1)[non_zero_indices] = tensor.view(-1)[non_zero_indices]
    return sparse_tensor
