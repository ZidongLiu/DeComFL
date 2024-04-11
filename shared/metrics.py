import torch


def accuracy(output: torch.tensor, target: torch.tensor) -> float:
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


class TensorMetric(object):
    def __init__(self, name):
        self.name = name
        self.tensor = None
        self.n = 0

    def reset(self):
        self.sum = None
        self.n = 0

    def update(self, update_tensor: torch.tensor):
        # self.sum += torch.allreduce(val.detach().cpu(), name=self.name)
        if self.tensor is None:
            self.tensor = update_tensor
        else:
            self.tensor += update_tensor

        self.n += 1

    @property
    def avg(self) -> float:
        return self.tensor / self.n


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
