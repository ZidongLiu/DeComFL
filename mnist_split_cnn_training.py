import torch
import torch.nn as nn
import torchvision

# Define transforms
import torchvision.transforms as transforms

from models.splitted_simple_cnn.splitted_simple_cnn import SimpleCNNSub1, SimpleCNNSub2
from optimizers.perturbation_direction_descent import PDD
from tqdm import tqdm


learn_rate = 1e-8
mu = 1e-4

model1 = SimpleCNNSub1()
pdd1 = PDD(model1.parameters())

model2 = SimpleCNNSub2()
pdd2 = PDD(model2.parameters())


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

# Load CIFAR-10 dataset
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=False, transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)


criterion = nn.CrossEntropyLoss()


batch_size = 2
n_round = 10000
eval_iteration = 500
train_update_iteration = 10

if __name__ == "__main__":

    with tqdm(total=n_round, desc="Training:") as t:
        with torch.no_grad():
            running_loss = 0.0
            running_accuracy = 0.0
            for cur_round, data in enumerate(trainloader):
                (images, labels) = data

                # model 1
                original_out_1 = model1(images)
                pdd1.apply_perturbation()
                perturbed_out_1 = model1(images)

                # model 2 and calulate loss and grad
                original_out_2 = model2(original_out_1)
                pdd2.apply_perturbation()
                perturbed_out_2 = model2(perturbed_out_1)

                original_loss = criterion(original_out_2, labels)
                perturbed_loss = criterion(perturbed_out_2, labels)
                grad = pdd2.calculate_grad(perturbed_loss, original_loss)

                # update model
                pdd2.step(grad)
                pdd1.step(grad)

                running_loss += original_loss.item()
                running_accuracy += (
                    (original_out_2.argmax(dim=1) == labels).float().mean().item()
                )
                if cur_round % train_update_iteration == (train_update_iteration - 1):
                    t.set_postfix(
                        {
                            "Loss": running_loss / train_update_iteration,
                            "Accuracy": running_accuracy / train_update_iteration,
                        }
                    )
                    t.update(train_update_iteration)
                    running_loss = 0.0
                    running_accuracy = 0.0
