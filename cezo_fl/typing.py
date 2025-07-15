from typing import Callable, TypeAlias

import torch

CriterionType: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
