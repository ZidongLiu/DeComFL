import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from config import get_params


def use_device(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        print("----- Using cuda -----")
        kwargs = (
            {"num_workers": args.num_workers, "pin_memory": True, "shuffle": True}
            if use_cuda
            else {}
        )
        return torch.device("cuda"), kwargs
    elif use_mps:
        print("----- Using mps -----")
        return torch.device("mps"), {}
    else:
        print("----- Using cpu -----")
        return torch.device("cpu"), {}


def preprocess(args):
    if args.dataset == "mnist":
        device, kwargs = use_device(args)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.train_batch_size, **kwargs
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, **kwargs
        )
    elif args.dataset == "cifar10":
        device, kwargs = use_device(args)
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.train_batch_size, **kwargs
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, **kwargs
        )

    return device, train_loader, test_loader
