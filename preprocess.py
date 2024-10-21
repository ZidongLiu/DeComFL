import torch
from torch.utils.data.dataset import Subset
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from transformers import AutoTokenizer

from cezo_fl.fl_helpers import get_client_name
from cezo_fl.util.data_split import dirichlet_split
from cezo_fl.util.dataset import ShakeSpeare
from cezo_fl.util.language_utils import (
    LM_DATASET_MAP,
    LM_TEMPLATE_MAP,
    SUPPORTED_LLM,
    CustomLMDataset,
    LmTask,
    get_collate_fn,
)


def use_device(args) -> tuple[dict[str, torch.device], dict]:
    num_clients = args.num_clients

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        num_gpu = torch.cuda.device_count()
        print(f"----- Using cuda count: {num_gpu} -----")
        # num_workers will make dataloader very slow especially when number clients is large
        # Do not shuffle shakespeare
        kwargs = {"pin_memory": True, "shuffle": args.dataset != "shakespeare"}
        server_device = {"server": torch.device("cuda:0")}
        client_devices = {
            get_client_name(i): torch.device(f"cuda:{(i+1) % num_gpu}") for i in range(num_clients)
        }
    elif use_mps:
        print("----- Using mps -----")
        print("----- Forcing model_dtype = float32 -----")
        args.model_dtype = "float32"
        kwargs = {}
        server_device = {"server": torch.device("mps")}
        client_devices = {get_client_name(i): torch.device("mps") for i in range(num_clients)}
    else:
        print("----- Using cpu -----")
        print("----- Forcing model_dtype = float32 -----")
        args.model_dtype = "float32"
        kwargs = {}
        server_device = {"server": torch.device("cpu")}
        client_devices = {get_client_name(i): torch.device("cpu") for i in range(num_clients)}

    return server_device | client_devices, kwargs


def preprocess(
    args,
) -> tuple[dict[str, torch.device], list[torch.utils.data.DataLoader], torch.utils.data.DataLoader]:
    device_map, kwargs = use_device(args)
    if args.dataset == "mnist":
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
        train_dataset = ShakeSpeare(train=True)
        test_dataset = ShakeSpeare(train=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, **kwargs
        )
    elif args.dataset in LM_TEMPLATE_MAP.keys():
        if args.dataset == LmTask.sst2.name:
            max_length = 32
        else:
            max_length = 2048

        dataset = load_dataset(LM_DATASET_MAP[args.dataset], args.dataset)
        raw_train_dataset = dataset["train"]
        raw_test_dataset = dataset["validation"]

        model_name = SUPPORTED_LLM[args.large_model]
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncate_side="left"
        )
        template = LM_TEMPLATE_MAP[args.dataset]()
        encoded_train_texts = list(map(template.verbalize, raw_train_dataset))
        encoded_test_texts = list(map(template.verbalize, raw_test_dataset))

        train_dataset = CustomLMDataset(encoded_train_texts, tokenizer, max_length=max_length)
        test_dataset = CustomLMDataset(encoded_test_texts, tokenizer, max_length=max_length)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=True,
            collate_fn=get_collate_fn(tokenizer, max_length),
        )

    else:
        raise Exception(f"Dataset {args.dataset} is not supported")

    # already updated at main function
    num_clients = args.num_clients
    splitted_train_sets: list[DatasetSplit] | list[Subset]
    if args.dataset == "shakespeare":
        dict_users = train_dataset.get_client_dic()
        splitted_train_sets = [
            DatasetSplit(train_dataset, dict_users[client_idx]) for client_idx in range(num_clients)
        ]
    elif args.dataset in LM_TEMPLATE_MAP.keys():
        if args.iid:
            generator = torch.Generator().manual_seed(args.seed)
            splitted_train_sets = torch.utils.data.random_split(
                train_dataset,
                get_random_split_chunk_length(len(train_dataset), num_clients),
                generator=generator,
            )
        else:
            labels = list(map(lambda x: x["label"], raw_train_dataset))
            splitted_train_sets = dirichlet_split(
                train_dataset, labels, num_clients, args.dirichlet_alpha, args.seed
            )
    else:
        generator = torch.Generator().manual_seed(args.seed)
        splitted_train_sets = torch.utils.data.random_split(
            train_dataset,
            get_random_split_chunk_length(len(train_dataset), num_clients),
            generator=generator,
        )
    splitted_train_loaders = []
    for i in range(num_clients):
        if args.dataset in LM_TEMPLATE_MAP.keys():
            dataloader = torch.utils.data.DataLoader(
                splitted_train_sets[i],
                batch_size=args.train_batch_size,
                shuffle=True,
                collate_fn=get_collate_fn(tokenizer, max_length),
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                splitted_train_sets[i], batch_size=args.train_batch_size, **kwargs
            )
        splitted_train_loaders.append(dataloader)
    return device_map, splitted_train_loaders, test_loader


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
