import torch
import torchvision
import torchvision.transforms as transforms
import json
from typing import Union
from shared.dataset import ShakeSpeare


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


def use_sparsity_dict(args, model_name: str) -> Union[dict[str, float], None]:
    if args.sparsity_file is None:
        print("Sparsity Dict: ", None)
        return None

    with open(args.sparsity_file, "r") as file:
        sparsity_data = json.load(file)

    sparsity_data_model = sparsity_data["model_name"]
    if sparsity_data_model != model_name:
        raise Exception(
            f"Sparsity file is generated using {sparsity_data_model}, "
            + f"while current specified model is {model_name}"
        )

    print("Sparsity Dict: ", sparsity_data["sparsity_dict"])
    return sparsity_data["sparsity_dict"]


def preprocess(args) -> tuple[str, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if args.dataset == "mnist":
        device, kwargs = use_device(args)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, **kwargs
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, **kwargs
        )
    elif args.dataset == "cifar10":
        device, kwargs = use_device(args)
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, **kwargs
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, **kwargs
        )
    elif args.dataset == "fashion":
        device, kwargs = use_device(args)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        train_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, **kwargs
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, **kwargs
        )
    elif args.dataset == "shakespeare":
        device, kwargs = use_device(args)
        train_dataset = ShakeSpeare(train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, **kwargs
        )
        test_dataset = ShakeSpeare(train=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )
    else:
        raise Exception(f"Dataset {args.dataset} is not supported")
    return device, train_loader, test_loader


def preprocess_cezo_fl(
    args,
) -> tuple[str, list[torch.utils.data.DataLoader], torch.utils.data.DataLoader]:
    if args.dataset == "mnist":
        device, kwargs = use_device(args)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, **kwargs
        )
    elif args.dataset == "cifar10":
        device, kwargs = use_device(args)
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, **kwargs
        )
    elif args.dataset == "fashion":
        device, kwargs = use_device(args)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        train_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, **kwargs
        )
    elif args.dataset == "shakespeare":
        device, kwargs = use_device(args)
        train_dataset = ShakeSpeare(train=True)
        test_dataset = ShakeSpeare(train=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )
    else:
        raise Exception(f"Dataset {args.dataset} is not supported")

    # already updated at main function
    num_clients = args.num_clients
    if args.dataset == "shakespeare":
        dict_users = train_dataset.get_client_dic()
        splitted_train_sets = [
            DatasetSplit(train_dataset, dict_users[client_idx]) for client_idx in range(num_clients)
        ]
    else:
        generator = torch.Generator().manual_seed(args.seed)
        splitted_train_sets = torch.utils.data.random_split(
            train_dataset,
            get_random_split_chunk_length(len(train_dataset), num_clients),
            generator=generator,
        )
    splitted_train_loaders = []
    for i in range(num_clients):
        splitted_train_loaders.append(
            torch.utils.data.DataLoader(
                splitted_train_sets[i], batch_size=args.train_batch_size, **kwargs
            )
        )
    return device, splitted_train_loaders, test_loader


class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_random_split_chunk_length(total_length: int, num_split: int) -> list[int]:
    int_len = total_length // num_split
    rem = total_length % num_split

    ret_base = [int_len] * num_split
    for i in range(rem):
        ret_base[i] += 1

    return ret_base
