import numpy as np
import json
import matplotlib.pyplot as plt
import glob
import pickle
import os
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

def json_to_pkl():
    for data_dir in data_dirs:
        data_name = data_dir.split("\\")[-3]
        print(data_name)
        pkl_name = hand_data_dir+data_name+".pkl"
        if os.path.isfile(pkl_name):
            pass
        else:
            dir_hand_seq = {}
            hand_jsons = glob.glob(data_dir + "*.json")
            hand_found = 0
            for hand_json in hand_jsons:
                json_num = int(hand_json[hand_json.find('_hd')+3:hand_json.find('.json')])

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
            file_object = open(hand_data_dir+data_name+".pkl", 'wb')
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
                        if ("right_hand" not in cur_id_data[t_index].keys()) or (0 in cur_id_data[t_index]["right_hand"]['landmarks']):
                            bad_keys.append(t_index)
                    for bad_key in bad_keys:
                        del cur_id_data[bad_key]
                    frame_nums = sorted(cur_id_data.keys())
                    if len(frame_nums) == 0:
                        continue
                    binary_array = np.zeros(frame_nums[-1]+1, dtype='int')
                    binary_array[frame_nums] = 1
                    zero_indices = np.where(binary_array == 0)[0]
                    for i in range(len(zero_indices)-1):
                        num_elem = zero_indices[i+1] - (zero_indices[i]+1)
                        if num_elem > 0:
                            current_array = np.zeros((num_elem, 63))
                            for j in range(zero_indices[i]+1, zero_indices[i+1]):
                                current_array[j-(zero_indices[i]+1), :] = cur_id_data[j]["right_hand"]['landmarks']
                            all_sequences.append(current_array)
            print(len(all_sequences))
            print(np.mean([seq.shape[0] for seq in all_sequences]))
        file_object = open(all_sequences_name, 'wb')
        pickle.dump(all_sequences, file_object)
        file_object.close()

def seq_to_tensor():
    if os.path.isfile(all_sequences_name):
        print("Loading sequences...")
        file_object = open(all_sequences_name, 'rb')
        all_sequences = pickle.load(file_object)
        file_object.close()
        print("Hand sequences loaded.")
    else:
        return
    seqs = [seq for seq in all_sequences if seq.shape[0] >= 10]
    seq_lengths = [seq.shape[0] for seq in seqs]
    def length_key(seq):
        return seq.shape[0]
    seqs = sorted(seqs, key=length_key, reverse=True)
    seq_lengths = [seq.shape[0] for seq in seqs]
    max_length = np.max(seq_lengths)
    print(seq_lengths)
    buffered_seqs = []
    out_np_array = np.zeros((max_length, 0, 63))
    for seq in seqs:
        new_seq = np.zeros((max_length, 63))
        new_seq[0:seq.shape[0], :] = seq
        new_seq = np.expand_dims(new_seq, 1)
        buffered_seqs.append(new_seq)
    print(len(buffered_seqs))


if __name__ == "__main__":
    #json_to_pkl()
    #fix_data()
    #pkl_to_seq()
    seq_to_tensor()