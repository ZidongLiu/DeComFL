import torch
import torch.nn as nn
import torchvision

# Define transforms
import torchvision.transforms as transforms
from models.mnist.simple_cnn import MnistSimpleCNN
from tqdm import tqdm
from tensorboardX import SummaryWriter
from os import path
from shared.model_helpers import get_current_datetime_str
from shared.metrics import Metric, accuracy
from optimizers.perturbation_direction_descent import PDD
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

model = MnistSimpleCNN()
criterion = nn.CrossEntropyLoss()

pdd = PDD(
    model.parameters(),
    lr=args.lr,
    mu=args.mu,
    grad_estimate_method=args.grad_estimate_method,
)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

# Load Mnist dataset
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.train_batch_size,
    **kwargs,
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
)


if __name__ == "__main__":

    trainset_len = len(train_loader)
    total_steps = args.epoch * trainset_len
    tensorboard_sub_folder = (
        f"pdd-{args.grad_estimate_method}-{get_current_datetime_str()}"
    )
    writer = SummaryWriter(path.join(tensorboard_path, tensorboard_sub_folder))

    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    eval_loss = Metric("Eval loss")
    eval_accuracy = Metric("Eval accuracy")

    with tqdm(total=total_steps, desc="Training:") as t, torch.no_grad():
        for epoch_idx in range(args.epoch):
            for train_batch_idx, data in enumerate(train_loader):
                trained_iteration = epoch_idx * trainset_len + train_batch_idx
                #
                (images, labels) = data
                # update models
                pdd.perturb_and_step(images, labels, model, criterion)
                # pred
                pred = model(images)
                #
                train_loss.update(criterion(pred, labels))
                train_accuracy.update(accuracy(pred, labels))

                if train_batch_idx % args.train_update_iteration == (
                    args.train_update_iteration - 1
                ):
                    t.set_postfix(
                        {
                            "Loss": train_loss.avg,
                            "Accuracy": train_accuracy.avg,
                        }
                    )

                    writer.add_scalar("Loss/train", train_loss.avg, trained_iteration)
                    writer.add_scalar(
                        "Accuracy/train", train_accuracy.avg, trained_iteration
                    )
                    t.update(args.train_update_iteration)

                    train_loss.reset()
                    train_accuracy.reset()

                if train_batch_idx % args.eval_iteration == (args.eval_iteration - 1):
                    for test_idx, data in enumerate(test_loader):
                        (test_images, test_labels) = data
                        pred = model(test_images)
                        eval_loss.update(criterion(pred, test_labels))
                        eval_accuracy.update(accuracy(pred, test_labels))

                    writer.add_scalar("Loss/test", eval_loss.avg, trained_iteration)
                    writer.add_scalar(
                        "Accuracy/test", eval_accuracy.avg, trained_iteration
                    )
                    print("")
                    print(
                        f"Evaluation(round {trained_iteration}): Eval Loss:{eval_loss.avg=:.4f}, Accuracy: {eval_accuracy.avg=: .4f}"
                    )
                    eval_loss.reset()
                    eval_accuracy.reset()

    writer.close()
