import torch
import torch.nn as nn
import torchvision

# Define transforms
import torchvision.transforms as transforms

from models.splitted_simple_cnn.splitted_simple_cnn import SimpleCNNSub1, SimpleCNNSub2
from optimizers.pdd_2_models import PDD2Models

from tqdm import tqdm
from tensorboardX import SummaryWriter
from os import path
from shared.model_helpers import get_current_datetime_str
from shared.metrics import Metric, accuracy

tensorboard_path = "./tensorboards/split_simple_cnn"

learning_rate = 1e-4
mu = 1e-4

model1 = SimpleCNNSub1()
model2 = SimpleCNNSub2()
criterion = nn.CrossEntropyLoss()

pdd_for_2_models = PDD2Models(model1, model2, learning_rate, mu, criterion)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

# Load Mnist dataset
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=False, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=False, transform=transform
)

test_loader = torch.utils.data.DataLoader(testset, batch_size=4, num_workers=2)


n_epoch = 10
eval_iteration = 5000
train_update_iteration = 100


if __name__ == "__main__":

    trainset_len = len(train_loader)
    total_steps = n_epoch * trainset_len
    tensorboard_sub_folder = f"{n_epoch}-{get_current_datetime_str()}"
    writer = SummaryWriter(path.join(tensorboard_path, tensorboard_sub_folder))

    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    eval_loss = Metric("Eval loss")
    eval_accuracy = Metric("Eval accuracy")

    with tqdm(total=total_steps, desc="Training:") as t:
        with torch.no_grad():
            for epoch_idx in range(n_epoch):
                for train_batch_idx, data in enumerate(train_loader):
                    trained_iteration = epoch_idx * trainset_len + train_batch_idx
                    #
                    (images, labels) = data

                    # update models
                    grad = pdd_for_2_models.step(images, labels)
                    # pred
                    pred = model2(model1(images))

                    train_loss.update(criterion(pred, labels))
                    train_accuracy.update(accuracy(pred, labels))
                    writer.add_scalar("Grad Scalar", grad, trained_iteration)

                    if train_batch_idx % train_update_iteration == (
                        train_update_iteration - 1
                    ):
                        t.set_postfix(
                            {
                                "Loss": train_loss.avg,
                                "Accuracy": train_accuracy.avg,
                            }
                        )

                        writer.add_scalar(
                            "Loss/train", train_loss.avg, trained_iteration
                        )
                        writer.add_scalar(
                            "Accuracy/train", train_accuracy.avg, trained_iteration
                        )
                        t.update(train_update_iteration)

                        train_loss.reset()
                        train_accuracy.reset()

                    if train_batch_idx % eval_iteration == (eval_iteration - 1):
                        for test_idx, data in enumerate(test_loader):
                            (test_images, test_labels) = data
                            pred = model2(model1(test_images))
                            eval_loss.update(criterion(pred, test_labels))
                            eval_accuracy.update(accuracy(pred, test_labels))

                        writer.add_scalar("Loss/test", eval_loss.avg, trained_iteration)
                        writer.add_scalar(
                            "Accuracy/test", eval_accuracy.avg, trained_iteration
                        )
                        print("")
                        print(f"Evaluation(round {trained_iteration})")
                        print(
                            f"Eval Loss:{eval_loss.avg:.4f}"
                            + f" Accuracy: {eval_accuracy.avg: .4f}"
                        )
                        eval_loss.reset()
                        eval_accuracy.reset()

    writer.close()
