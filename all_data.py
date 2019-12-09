import numpy as np
import json
import matplotlib.pyplot as plt
import glob
import pickle
import os
import torch
from viz import visualize_hands

plt.rcParams['image.interpolation'] = 'nearest'

# Setup paths
data_path = '../../../hand_data/'
data_dirs = glob.glob(data_path + "*/hdHand3d/")
print(data_dirs)
total_hands = 0
hand_sequences = {}
hand_data_dir = data_path + "pkls/"
hand_data_dir2 = data_path + "pkls3/"
all_sequences_name = data_path + "/hand_sequences.pkl"
sequence_tensor_name = data_path + "/sequence_tensor.pkl"


def json_to_pkl():
    for data_dir in data_dirs:
        data_name = data_dir.split("\\")[-3]
        print(data_name)
        pkl_name = hand_data_dir + data_name + ".pkl"
        if os.path.isfile(pkl_name):
            pass
        else:
            dir_hand_seq = {}
            hand_jsons = glob.glob(data_dir + "*.json")
            hand_found = 0
            for hand_json in hand_jsons:
                json_num = int(hand_json[hand_json.find('_hd') + 3:hand_json.find('.json')])

                with open(hand_json) as dfile:
                    hframe = json.load(dfile)
                for person in hframe['people']:
                    if "id" in person.keys():
                        if 'right_hand' in person.keys():
                            cur_id = str(person["id"]) + 'r'
                            if cur_id not in dir_hand_seq.keys():
                                dir_hand_seq[cur_id] = {}
                            dir_hand_seq[cur_id][json_num] = person["right_hand"]
                        if 'left_hand' in person.keys():
                            cur_id = str(person["id"]) + 'l'
                            if cur_id not in dir_hand_seq.keys():
                                dir_hand_seq[cur_id] = {}
                            dir_hand_seq[cur_id][json_num] = person["left_hand"]
            file_object = open(hand_data_dir + data_name + ".pkl", 'wb')
            pickle.dump(dir_hand_seq, file_object)
            file_object.close()


def fix_data():
    pkl_dirs = glob.glob(hand_data_dir + "*.pkl")
    for pkl_dir in pkl_dirs:
        print(pkl_dir)
        pkl_name = pkl_dir.split('\\')[-1]
        file_object = open(pkl_dir, 'rb')
        cur_hand_seq = pickle.load(file_object)
        file_object.close()
        old_ids = list(cur_hand_seq.keys())
        for id in old_ids:
            cur_id_data = cur_hand_seq[id]
            new_id = id[0:-1]
            if new_id not in cur_hand_seq:
                cur_hand_seq[new_id] = {}
            else:
                continue
            for t_index in cur_id_data.keys():
                cur_hand_seq[new_id][t_index] = cur_hand_seq[id][t_index]
        for id in old_ids:
            del cur_hand_seq[id]
        file_object = open(hand_data_dir2 + pkl_name, 'wb')
        pickle.dump(cur_hand_seq, file_object)
        file_object.close()


def pkl_to_seq():
    if os.path.isfile(all_sequences_name):
        print("Loading sequences...")
        file_object = open(all_sequences_name, 'rb')
        all_sequences = pickle.load(file_object)
        file_object.close()
        print("Hand sequences loaded.")
    else:
        pkl_dirs = glob.glob(hand_data_dir + "*.pkl")
        all_sequences = []
        for pkl_dir in pkl_dirs:
            print(pkl_dir)
            file_object = open(pkl_dir, 'rb')
            cur_hand_seq = pickle.load(file_object)
            file_object.close()
            for id in cur_hand_seq:
                if '-1' not in id:
                    cur_id_data = cur_hand_seq[id]
                    bad_keys = []
                    for t_index in cur_id_data.keys():
                        if ("right_hand" not in cur_id_data[t_index].keys()) or (
                                0 in cur_id_data[t_index]["right_hand"]['landmarks']):
                            bad_keys.append(t_index)
                    for bad_key in bad_keys:
                        del cur_id_data[bad_key]
                    frame_nums = sorted(cur_id_data.keys())
                    if len(frame_nums) == 0:
                        continue
                    binary_array = np.zeros(frame_nums[-1] + 1, dtype='int')
                    binary_array[frame_nums] = 1
                    zero_indices = np.where(binary_array == 0)[0]
                    for i in range(len(zero_indices) - 1):
                        num_elem = zero_indices[i + 1] - (zero_indices[i] + 1)
                        if num_elem > 0:
                            current_array = np.zeros((num_elem, 63))
                            for j in range(zero_indices[i] + 1, zero_indices[i + 1]):
                                current_array[j - (zero_indices[i] + 1), :] = cur_id_data[j]["right_hand"]['landmarks']
                            all_sequences.append(current_array)
            print(len(all_sequences))
            print(np.mean([seq.shape[0] for seq in all_sequences]))
        file_object = open(all_sequences_name, 'wb')
        pickle.dump(all_sequences, file_object)
        file_object.close()


def seq_to_tensor():
    if os.path.isfile(sequence_tensor_name):
        print("Loading sequences...")
        file_object = open(sequence_tensor_name, 'rb')
        out_np_array, seq_lengths = pickle.load(file_object)
        file_object.close()
        print("Seq tensor loaded.")

    else:
        print("Loading sequences...")
        file_object = open(all_sequences_name, 'rb')
        all_sequences = pickle.load(file_object)
        file_object.close()
        print("Hand sequences loaded.")

        seqs = [seq for seq in all_sequences if seq.shape[0] >= 10]
        shorter_seqs = []
        for seq in seqs:
            split_array = np.array_split(seq, int(seq.shape[0] / 180) + 1, axis=0)
            shorter_seqs.extend(split_array)
        seqs = shorter_seqs

        def length_key(seq):
            return seq.shape[0]

        seqs = sorted(seqs, key=length_key, reverse=True)
        seq_lengths = [seq.shape[0] for seq in seqs]
        max_length = np.max(seq_lengths)
        buffered_seqs = []
        num_seqs = len(seqs)
        out_np_array = np.zeros((max_length, num_seqs, 63))
        i = 0
        for seq in seqs:
            new_seq = np.zeros((max_length, 63))
            new_seq[0:seq.shape[0], :] = seq
            new_seq = np.expand_dims(new_seq, 1)
            buffered_seqs.append(new_seq)
            out_np_array[0:max_length, i:i + 1, 0:63] = new_seq
            i += 1

        file_object = open(sequence_tensor_name, 'wb')
        pickle.dump((out_np_array, seq_lengths), file_object)
        file_object.close()
    return out_np_array, seq_lengths

time_ahead = 2

def generate_input():
    input_tensor, seq_lengths = seq_to_tensor()
    for i in range(time_ahead):
        input_tensor[[s - time_ahead for s in seq_lengths], list(range(len(seq_lengths))), :] = 0
    input_tensor = input_tensor[0:-time_ahead, :, :]
    return input_tensor, seq_lengths


def generate_output():
    output_tensor, seq_lengths = seq_to_tensor()
    return output_tensor[time_ahead:, :, :], seq_lengths

def subtract_wrist(cur_tensor):
    wrists = cur_tensor[:,:,0:3]
    wrist_rep = wrists.repeat(1, 1, int(cur_tensor.shape[2]/3))
    out_tensor = cur_tensor - wrist_rep
    return out_tensor

hand_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
                           [0, 5], [5, 6], [6, 7], [7, 8],
                           [0, 9], [9, 10], [10, 11], [11, 12],
                           [0, 13], [13, 14], [14, 15], [15, 16],
                           [0, 17], [17, 18], [18, 19], [19, 20]])

angle_indices = [[0,1,2], [1,2,3], [2,3,4],
                 [0,5,6], [5,6,7], [6,7,8],
                 [0,9,10], [9,10,11], [10,11,12],
                 [0,13,14], [13,14,15], [14,15,16],
                 [0,17,18], [17,18,19], [18,19,20],
                 [2,0,9], [5,0,9], [9,0,13], [13,0,17]]

def get_angle_tensor(cur_tensor, angle_indices, seq_lengths):
    num_angles = len(angle_indices)
    new_tensor = torch.zeros(cur_tensor.shape[0], cur_tensor.shape[1], num_angles, device=cur_tensor.device, dtype=cur_tensor.dtype)
    i = 0
    for points in angle_indices:
        p = [0,0,0]
        for j in range(3):
            ind = points[j]
            p[j] = cur_tensor[:, :, [ind*3, ind*3+1,ind*3+2]]
        v0 = -(p[0] - p[1]) + .0000001
        v1 = p[2] - p[1] + .0000001
        norm0 = torch.norm(v0, p=2, dim=2)
        norm1 = torch.norm(v1, p=2, dim=2)
        dot0 = torch.sum(v0*v1, dim=2)
        angle = torch.acos(.99999*dot0/(norm0*norm1))
        new_tensor[:,:,i] = angle
        i = i+1
    for i in range(len(seq_lengths)):
        seq_length = seq_lengths[i] - time_ahead
        new_tensor[seq_length:, i, :] = 0
    return new_tensor

if __name__ == "__main__":
    pass
    # json_to_pkl()
    # fix_data()
    # pkl_to_seq()
    #seq_to_tensor()
    #printgenerate_input()[0][0:20,41,:])
    print(generate_output()[0][0:4, 1, :])
