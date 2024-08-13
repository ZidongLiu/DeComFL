import torch


def mean(num_sample_clients: int, local_grad_scalar_list):
    grad_scalar: list[torch.Tensor] = []
    for each_client_update in zip(*local_grad_scalar_list):
        grad_scalar.append(sum(each_client_update).div_(num_sample_clients))
    return grad_scalar


def median(local_grad_scalar_list):
    grad_scalar: list[torch.Tensor] = []
    for each_client_update in zip(*local_grad_scalar_list):
        each_client_tensor = torch.stack(each_client_update)
        median = torch.median(each_client_tensor, dim=0).values
        grad_scalar.append(median)
    return grad_scalar


# delete the biggest f and smallest f elements,
# and then calculate the average of the rest of elements
def trim(num_sample_clients: int, local_grad_scalar_list, f=1):
    grad_scalar: list[torch.Tensor] = []
    for each_client_update in zip(*local_grad_scalar_list):
        each_client_tensor = torch.stack(each_client_update)
        sorted_tensor, _ = torch.sort(each_client_tensor, dim=0)
        trimmed_tensor = sorted_tensor[f : num_sample_clients - f]
        trimmed_mean = trimmed_tensor.mean(dim=0)
        grad_scalar.append(trimmed_mean)
    return grad_scalar


def krum():
    return
