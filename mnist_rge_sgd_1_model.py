import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Define transforms
import torchvision.transforms as transforms

from models.mnist.simple_cnn import MnistSimpleCNN

from tqdm import tqdm
from tensorboardX import SummaryWriter
from os import path
from shared.model_helpers import get_current_datetime_str
from shared.metrics import Metric, accuracy
from optimizers.rge_sgd import RGE_SGD
from config import get_params


args = get_params("").parse_args()
torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()
if use_cuda:
    device = torch.device("cuda")
    print("----- Using cuda -----")
elif use_mps:
    device = torch.device("mps")
    print("----- Using mps -----")
else:
    device = torch.device("cpu")
    print("----- Using cpu -----")

kwargs = (
    {"num_workers": args.num_workers, "pin_memory": True, "shuffle": True}
    if use_cuda
    else {}
)


tensorboard_path = "./tensorboards/1_model"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Load Mnist dataset
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


model, criterion = MnistSimpleCNN(), nn.CrossEntropyLoss()

n_permutation = 1
rge_sgd = RGE_SGD(
    list(model.parameters()), lr=args.lr, mu=args.mu, n_permutation=n_permutation
)


def train_model(epoch: int) -> tuple[float, float]:
    model.train()
    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    with tqdm(total=len(train_loader), desc="Training:") as t, torch.no_grad():
        for train_batch_idx, (images, labels) in enumerate(train_loader):
            # update models
            rge_sgd.step(images, labels, model, criterion)
            pred = model(images)

            train_loss.update(criterion(pred, labels))
            train_accuracy.update(accuracy(pred, labels))
            t.set_postfix({"Loss": train_loss.avg, "Accuracy": train_accuracy.avg})
            t.update(1)
    return train_loss.avg, train_accuracy.avg


def eval_model(epoch: int) -> tuple[float, float]:
    model.eval()
    eval_loss = Metric("Eval loss")
    eval_accuracy = Metric("Eval accuracy")
    with torch.no_grad():
        for test_idx, (images, labels) in enumerate(test_loader):
            pred = model(images)
            eval_loss.update(criterion(pred, labels))
            eval_accuracy.update(accuracy(pred, labels))
    print(
        f"Evaluation(round {epoch}): Eval Loss:{eval_loss.avg=:.4f}, "
        f"Accuracy: {eval_accuracy.avg * 100=: .2f}%"
    )
    return eval_loss.avg, eval_accuracy.avg


if __name__ == "__main__":

    tensorboard_sub_folder = (
        f"rge_sgd-{args.grad_estimate_method}-"
        + f"n_permutation-{n_permutation}-{get_current_datetime_str()}"
    )
    writer = SummaryWriter(path.join(tensorboard_path, tensorboard_sub_folder))
    for epoch in range(args.epoch):
        train_loss, train_accuracy = train_model(epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)

        eval_loss, eval_accuracy = eval_model(epoch)
        writer.add_scalar("Loss/test", eval_loss, epoch)
        writer.add_scalar("Accuracy/test", eval_accuracy, epoch)

    writer.close()
