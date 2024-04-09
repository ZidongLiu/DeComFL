import torch


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

    def update(self, val: float):
        # self.sum += torch.allreduce(val.detach().cpu(), name=self.name)
        self.sum += val
        self.n += 1

    @property
    def avg(self) -> float:
        return (self.sum / self.n).item()
