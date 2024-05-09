import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=64, num_layers=10, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x, None)
        out = self.fc(out[:, -1, :])
        return out


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)

        self.out = nn.Sequential(nn.Linear(64, 1))

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])

        return out


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.out = nn.Sequential(nn.Linear(64, 1))

    def forward(self, x):
        r_out, h = self.gru(x, None)
        out = self.out(r_out[:, -1, :])

        return out
