import torchvision
from shared.model_helpers import Net

model = Net()

mnist_train = torchvision.datasets.MNIST(root='\data', train=True, transform=None, download=True)
