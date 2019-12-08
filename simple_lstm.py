import torch
from torch import nn
import numpy as np
from all_data import *

class Simple_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Simple_LSTM,self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x,_ = self.lstm(x)
        x,_ = torch.nn.utils.rnn.pad_packed_sequence(x, total_length = 180-time_ahead)
        x = self.fc(x)
        return x

