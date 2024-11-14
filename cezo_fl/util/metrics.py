import torch


def accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


class Metric(object):
    def __init__(self, name: str):
        self.name = name
        self.sum = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def reset(self):
        self.sum = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, val: float | torch.Tensor):
        if isinstance(val, float):
            self.sum += val
        else:
            self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self) -> float:
        return (self.sum / self.n).item()
