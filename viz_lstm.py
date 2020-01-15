import simple_lstm
import torch
from torch import nn, optim
from simple_lstm import *
from all_data import *
import random
from viz import *
from forward_kinematics import *

angle_indices = [[0,1,2], [1,2,3], [2,3,4],
                 [0,5,6], [5,6,7], [6,7,8],
                 [0,9,10], [9,10,11], [10,11,12],
                 [0,13,14], [13,14,15], [14,15,16],
                 [0,17,18], [17,18,19], [18,19,20],
                 [1,0,9], [5,0,9], [9,0,13], [13,0,17]]
test_names = ["original_hands", "angle_with_recon", "recon_only"]
mod_list = [None, "angle_tensor", "recon_tensor"]
titles = ["Loss on Tracked Hands", "Loss on Reconstructed Hands with Angle Training", "Loss on Reconstructed Hands"]
ymaxes = [50,2,60]
epochs = [1000,600,2000]
data_index = 17

test_index = 2
test_name = test_names[test_index]
mods = mod_list[test_index]
title = titles[test_index]
ymax = ymaxes[test_index]
epoch = epochs[test_index]

file_object = open(test_name + "/losses_"+str(epoch) + ".pkl", 'rb')
losses = pickle.load(file_object)
file_object.close()

print(np.min([loss.cpu().numpy().item() for loss in losses[0]]))
print([loss.cpu().numpy().item() for loss in losses[1]])
print([loss.cpu().numpy().item() for loss in losses[2]])

plot_loss(losses[0:3], title = title, ymax=ymax)

model = torch.load(test_name +  "/model_" + str(epoch) + ".pt")
cuda = torch.device('cuda')
random.seed(0)

input_numpy, seq_lengths = generate_input()
target_numpy, _ = generate_output()

input_tensor = torch.from_numpy(input_numpy).float().to(cuda)
target_tensor = torch.from_numpy(target_numpy).float().to(cuda)

input_tensor = input_tensor
target_tensor = target_tensor

if mods == "angle_tensor":
    input_tensor = get_angle_tensor(input_tensor, angle_indices, seq_lengths)
    target_tensor = get_angle_tensor(target_tensor, angle_indices, seq_lengths)
elif mods == "recon_tensor":
    input_tensor = angle_tensor_to_hand(get_angle_tensor(input_tensor, angle_indices, seq_lengths))
    target_tensor = angle_tensor_to_hand(get_angle_tensor(target_tensor, angle_indices, seq_lengths))


num_points = len(seq_lengths)
shuffled_indices = list(range(len(seq_lengths)))
random.shuffle(shuffled_indices)
train_indices = shuffled_indices[0:int(num_points * .8)]
test_indices = shuffled_indices[int(num_points * .8):num_points]

with torch.no_grad():

    sorted_test = sorted(test_indices)
    test_all = input_tensor[:, sorted_test, :]
    what = test_all.cpu()
    target_test = target_tensor[:, sorted_test, :]
    lstm_in = torch.nn.utils.rnn.pack_padded_sequence(test_all,
                                                      [seq_lengths[t] - time_ahead for t in sorted_test])
    test_out = model(lstm_in)
    if  mods == "angle_tensor":
        test_out = angle_tensor_to_hand(test_out)
        target_test = angle_tensor_to_hand(target_test)
        test_all = angle_tensor_to_hand(test_all)
    vis_py = np.zeros((2,63))
    vis_py_test= test_out.cpu().numpy()[100:101, data_index, :]
    vis_py_target = target_test.cpu().numpy()[100:101, data_index, :]
    vis_py_in = test_all.cpu().numpy()[97:101, data_index, :]
    visualize_hands([vis_py_test, vis_py_target, vis_py_in])


