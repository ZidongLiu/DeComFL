import torch


def set_all_param_zero(model):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()


def set_all_grad_zero(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.zero_()


def set_flatten_model_back(model, x_flattern):
    with torch.no_grad():
        start = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p_extract = x_flattern[start : (start + p.numel())]
            p.set_(p_extract.view(p.shape).clone())
            if p.grad is not None:
                p.grad.zero_()
            start += p.numel()


def get_flatten_model_param(model):
    with torch.no_grad():
        return torch.cat([p.detach().view(-1) for p in model.parameters() if p.requires_grad])


def get_flatten_model_grad(model):
    with torch.no_grad():
        return torch.cat([p.grad.detach().view(-1) for p in model.parameters() if p.requires_grad])


def accuracy(output: torch.tensor, target: torch.tensor) -> float:
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.0)  # pylint: disable=not-callable
        self.n = torch.tensor(0.0)  # pylint: disable=not-callable

    def reset(self):
        self.sum = torch.tensor(0.0)  # pylint: disable=not-callable
        self.n = torch.tensor(0.0)  # pylint: disable=not-callable

    def update(self, val: float | torch.Tensor):
        if isinstance(val, float):
            self.sum += val
        else:
            self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self) -> float:
        return (self.sum / self.n).item()
