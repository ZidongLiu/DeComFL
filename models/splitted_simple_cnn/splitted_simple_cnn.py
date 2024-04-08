import torch.nn as nn


class SimpleCNNSub1(nn.Module):

    def __init__(self, n_channel=1):
        super(SimpleCNNSub1, self).__init__()

        self.n_channel = n_channel

        self.conv1 = nn.Conv2d(self.n_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        return x


class SimpleCNNSub2(nn.Module):

    def __init__(self, n_out_category=10):
        super(SimpleCNNSub2, self).__init__()
        self.n_out_category = n_out_category

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n_out_category)

    def forward(self, x):
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
