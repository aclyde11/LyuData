import torch
import torch.nn as nn
import torch.nn.functional as F


# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836943/pdf/MINF-37-na.pdf
class DockRegressor(nn.Module):
    def __init__(self, vocab_size, emb_size, max_len=150):
        super(DockRegressor, self).__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.RNN(emb_size, 128, dropout=0.05, num_layers=4)
        self.convnet = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 8, kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 8, kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        # self.lstm2 = nn.GRU(8, 8, dropout=0.05, num_layers=1, batch_first=True)

        self.linear1 = nn.Sequential(nn.Linear(128 * max_len, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256,1))
        self.linear2 = nn.Sequential(nn.Linear(128 * max_len, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256,1))


    # pass x as a pack padded sequence please.
    def forward(self, x):
        # do stuff to train
        batch_size = len(x)

        x = [self.emb(x_) for x_ in x]

        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

        x, _ = self.lstm(x)

        x, _ = nn.utils.rnn.pad_packed_sequence(x, padding_value=0, total_length=self.max_len)
        x = x.permute((1, 2,0))
        x = self.convnet(x)
        # x = x.permute(0, 2, 1)
        # x, _ = self.lstm2(x)
        # x = x.permute((1,0, 2))
        x = x.reshape(batch_size, -1)
        return self.linear1(x), self.linear2(x)
