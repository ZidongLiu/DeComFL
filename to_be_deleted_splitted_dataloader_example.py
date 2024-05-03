import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

from torch.utils.data.distributed import DistributedSampler

num_clients = 5

samplers = []
dataloaders = []
for i in range(num_clients):
    sampler = DistributedSampler(trainset, num_replicas=num_clients, rank=i, shuffle=False, drop_last=False)
    dataloader = DataLoader(trainset, batch_size=128, pin_memory=True, num_workers=1, drop_last=False, shuffle=False, sampler=sampler)
    samplers += [sampler]
    dataloaders += [dataloader]



from torch.utils.data import random_split

subsets = random_split(trainset, [1/num_clients for _ in range(num_clients)])
subset_loaders = []
for subset in subsets:
    subset_loader = DataLoader(subset, batch_size=128, pin_memory=True, num_workers=1, shuffle=False)
    subset_loaders += [subset_loader]