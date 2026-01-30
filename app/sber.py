import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_features, hidden_size):
        super(Net, self).__init__()
        self.n_features = n_features
        self.ln1 = nn.Linear(n_features, hidden_size)
        self.act1 = nn.ReLU()
        self.ln2 = nn.Linear(hidden_size, 2)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.ln1(x)
        x = self.act1(x)
        x = self.ln2(x)
        x = self.act2(x)

        return x

