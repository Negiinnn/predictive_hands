import simple_lstm
import torch
from torch import nn, optim
from simple_lstm import *

input_dim = 21
hidden_dim = 5
eps = .001
model = Simple_LSTM(input_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
def loss_function(output, target, eps):
    1 - (output+eps)/(torch.abs(target)+eps)



for epoch in epochs:
    for data, target in training_data:
        model.zero_grad()
        out = model(data)
        loss = loss_function(out, target, eps)
        loss.backwards()
        optimizer.step()

with torch.no_grad():
    pass
    #check results



