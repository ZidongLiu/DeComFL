from torch import nn


# Used for Shakespeare dataset
class CharLSTM(nn.Module):
    model_name = "LSTM"

    def __init__(self):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
        # self.drop = nn.Dropout()
        self.out = nn.Linear(256, 80)

    def forward(self, x):
        x = self.embed(x)
        x, hidden = self.lstm(x)
        # x = self.drop(x)
        return self.out(x[:, -1, :])
