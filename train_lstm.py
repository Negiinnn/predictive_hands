import simple_lstm
import torch
from torch import nn, optim
from simple_lstm import *
from all_data import *
import random

input_dim = 63
hidden_dim = 256
eps = .01
batch_size = 32
epochs = 10000
lr = 1e-4
cuda = torch.device('cuda')
random.seed(0)
test_name = "angle_tensor"
mods = "angle_tensor"
angle_indices = [[0,1,2], [1,2,3], [2,3,4],
                 [0,5,6], [5,6,7], [6,7,8],
                 [0,9,10], [9,10,11], [10,11,12],
                 [0,13,14], [13,14,15], [14,15,16],
                 [0,17,18], [17,18,19], [18,19,20],
                 [1,0,9], [5,0,9], [9,0,13], [13,0,17]]
if mods == "angle_tensor":
    input_dim = 19

#def loss_function(output, target):
#    return torch.mean((torch.abs(target-output) + eps) / (torch.abs(target) + eps))
loss_function = torch.nn.MSELoss()

if __name__ == "__main__":
    loss_1 = []
    loss_2 = []
    loss_3 = []
    loss_4 = []
    model = Simple_LSTM(input_dim, hidden_dim).to(cuda)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    input_numpy, seq_lengths = generate_input()
    target_numpy, _ = generate_output()
    input_tensor = torch.from_numpy(input_numpy).float().to(cuda)
    target_tensor = torch.from_numpy(target_numpy).float().to(cuda)

    if mods == 'subtract_wrist':
        input_tensor = subtract_wrist(input_tensor)
        target_tensor = subtract_wrist(target_tensor)
    elif mods == 'angle_tensor':
        input_tensor = get_angle_tensor(input_tensor, angle_indices, seq_lengths)
        target_tensor = get_angle_tensor(target_tensor, angle_indices, seq_lengths)

    num_points = len(seq_lengths)
    shuffled_indices = list(range(len(seq_lengths)))
    random.shuffle(shuffled_indices)
    train_indices = shuffled_indices[0:int(num_points * .8)]
    test_indices = shuffled_indices[int(num_points * .8):num_points]
    if not os.path.exists(test_name):
        os.makedirs(test_name)
    file_object = open(test_name + "/indices" + ".pkl", 'wb')
    pickle.dump((train_indices, test_indices), file_object)
    file_object.close()

    for epoch in range(epochs):
        cur_batches = random.shuffle(train_indices)
        num_batches = int(len(train_indices) / batch_size)
        train_loss = 0
        for i in range(num_batches):
           # print(i)
            if i == list(range(num_batches))[-1]:
                sorted_batch = sorted(train_indices[i * batch_size:])
            else:
                sorted_batch = sorted(train_indices[i * batch_size: (i + 1) * batch_size])
            train_batch = input_tensor[:, sorted_batch, :]
            target_batch = target_tensor[:, sorted_batch, :]
            lstm_in = torch.nn.utils.rnn.pack_padded_sequence(train_batch, [seq_lengths[t] - time_ahead for t in sorted_batch])

            model.zero_grad()
            out = model(lstm_in)
            loss = loss_function(out, target_batch)
            with torch.no_grad():
                train_loss += loss
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            sorted_test = sorted(test_indices)
            test_all = input_tensor[:, sorted_test, :]
            target_test = target_tensor[:, sorted_test, :]
            lstm_in = torch.nn.utils.rnn.pack_padded_sequence(test_all,
                                                              [seq_lengths[t] - time_ahead for t in sorted_test])
            test_out = model(lstm_in)
            print("hello")
            print(loss_function(test_out, target_test))
            print(loss_function(test_out, test_all))
            print(loss_function(test_all, target_test))
            print(train_loss)
            loss_1.append(loss_function(test_out, target_test))
            loss_2.append(loss_function(test_out, test_all))
            loss_3.append(loss_function(test_all, target_test))
            loss_4.append(train_loss)
            if epoch % 50 == 0:
                torch.save(model, test_name + "/model_" + str(epoch) + ".pt")
                file_object = open(test_name + "/losses_" + str(epoch) + ".pkl", 'wb')
                pickle.dump((loss_1, loss_2, loss_3, loss_4), file_object)
                file_object.close()