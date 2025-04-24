import torch


def mean(local_grad_scalar_list: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    num_sample_clients = len(local_grad_scalar_list)
    grad_scalar: list[torch.Tensor] = []
    for each_local_step_update in zip(*local_grad_scalar_list):
        grad_scalar.append(sum(each_local_step_update).div_(num_sample_clients))
    return grad_scalar


def median(local_grad_scalar_list: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    grad_scalar: list[torch.Tensor] = []
    for each_local_step_update in zip(*local_grad_scalar_list):
        each_client_tensor = torch.stack(each_local_step_update)
        median = torch.median(each_client_tensor, dim=0).values
        grad_scalar.append(median)
    return grad_scalar


# delete the biggest f and smallest f elements,
# and then calculate the average of the rest of elements
def trim(local_grad_scalar_list: list[list[torch.Tensor]], f=1) -> list[torch.Tensor]:
    num_sample_clients = len(local_grad_scalar_list)
    assert 0 < 2 * f < len(local_grad_scalar_list)
    grad_scalar: list[torch.Tensor] = []
    for each_local_step_update in zip(*local_grad_scalar_list):
        each_client_tensor = torch.stack(each_local_step_update)
        sorted_tensor, _ = torch.sort(each_client_tensor, dim=0)
        trimmed_tensor = sorted_tensor[f : num_sample_clients - f]
        trimmed_mean = trimmed_tensor.mean(dim=0)
        grad_scalar.append(trimmed_mean)
    return grad_scalar


def score(grad: torch.Tensor, v: torch.Tensor, f: int = 1) -> float:
    grad = grad.view(-1, 1)
    distances = torch.sum((v - grad) ** 2, dim=0)
    sorted_distance, _ = torch.sort(distances)
    num_neighbours = v.shape[1] - 2 - f
    return torch.sum(sorted_distance[1 : (1 + num_neighbours)]).item()


def krum(local_grad_scalar_list: list[list[torch.Tensor]], f: int = 1) -> list[torch.Tensor]:
    assert 0 < 2 * f < len(local_grad_scalar_list)
    grad_scalar: list[torch.Tensor] = []
    for each_local_step_update in zip(*local_grad_scalar_list):
        v = torch.stack(each_local_step_update, dim=1)
        scores = torch.tensor([score(grad, v, f) for grad in v.t()])
        min_idx = int(torch.argmin(scores).item())
        grad_scalar.append(v[:, min_idx])
    return grad_scalar
