import torch
import torch.nn as nn
import torchvision

# Define transforms
import torchvision.transforms as transforms

from models.splitted_simple_cnn.splitted_simple_cnn import SimpleCNNSub1, SimpleCNNSub2
from optimizers.perturbation_direction_descent import PDD, PDD_training_loop


learning_rate = 1e-4
mu = 1e-4

model1 = SimpleCNNSub1()
pdd1 = PDD(model1.parameters(), lr=learning_rate, mu=mu)

model2 = SimpleCNNSub2()
pdd2 = PDD(model2.parameters(), lr=learning_rate, mu=mu)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

# Load Mnist dataset
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=False, transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=False, transform=transform
)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, num_workers=2)

criterion = nn.CrossEntropyLoss()


batch_size = 2
n_epoch = 10
eval_iteration = 5000
train_update_iteration = 100

if __name__ == "__main__":
    PDD_training_loop(
        model1,
        model2,
        pdd1,
        pdd2,
        criterion,
        trainloader,
        testloader,
        n_epoch,
        train_update_iteration,
        eval_iteration,
        "./models/splitted_simple_cnn/tensorboard",
    )
