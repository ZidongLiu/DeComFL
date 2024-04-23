import os
import torch
from torch import nn
import json
from config import get_params
from preprocess import preprocess
from pruning.model_prune import zoo_grasp_prune
from pruning.helpers import get_module_weight_sparsity

from models.cnn_mnist import CNN_MNIST

from models.cnn_cifar10 import CNN_CIFAR10
from models.resnet import ResNet18
from models.lenet import LeNet
from models.cnn_fashion import CNN_FMNIST

if __name__ == "__main__":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    parser = get_params()
    # has one more args than rge_main
    parser.add_argument("--sparsity", type=float, default=0.9)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device, train_loader, test_loader = preprocess(args)
    criterion = nn.CrossEntropyLoss()

    if args.dataset == "mnist":
        model = CNN_MNIST().to(device)
    elif args.dataset == "cifar10":
        model = ResNet18().to(device)
    elif args.dataset == "cifar10":
        model = LeNet().to(device)
    elif args.dataset == "fashion":
        model = CNN_FMNIST()

    model_name = model.model_name
    print(args.dataset, model_name)

    zoo_grasp_prune(
        model,
        ratio=args.sparsity,
        dataloader=train_loader,
        sample_per_classes=25,
        num_pert=192,
        mu=5e-3,
    )

    weight_sparsity_dict = get_module_weight_sparsity(model)

    os.makedirs(f"saved_sparsity/{args.dataset}", exist_ok=True)
    with open(
        f"saved_sparsity/{args.dataset}/zoo_grasp_{args.sparsity}_{model_name}.json",
        "w",
    ) as file:
        json.dump(
            {"model_name": model_name, "sparsity_dict": weight_sparsity_dict},
            file,
        )
