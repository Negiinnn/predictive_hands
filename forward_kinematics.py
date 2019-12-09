import numpy as np
import json
import matplotlib.pyplot as plt
import glob
import pickle
import os
import torch
from viz import visualize_hands

forward_dict = []
forward_dict.append({'finger_angles' : [0,1,2], 'sep_angle' : -15})
forward_dict.append({'finger_angles' : [3,4,5], 'sep_angle' : None})
forward_dict.append({'finger_angles' : [6,7,8], 'sep_angle' : 16})
forward_dict.append({'finger_angles' : [9,10,11], 'sep_angle' : 17})
forward_dict.append({'finger_angles' : [12,13,14], 'sep_angle' : 18})


angle_indices = [[0,1,2], [1,2,3], [2,3,4],
                 [0,5,6], [5,6,7], [6,7,8],
                 [0,9,10], [9,10,11], [10,11,12],
                 [0,13,14], [13,14,15], [14,15,16],
                 [0,17,18], [17,18,19], [18,19,20],
                 [1,0,9], [5,0,9], [9,0,13], [13,0,17]]

def angle_tensor_to_hand(cur_tensor):
    segment_length = 6
    new_tensor = torch.zeros(cur_tensor.shape[0], cur_tensor.shape[1], 63, device=cur_tensor.device,
                             dtype=cur_tensor.dtype)
    i = 0
    new_tensor[:, :, i * 3] = -1*segment_length
    #new_tensor[:, :, i * 3+2] = 2*segment_length/3
    i += 1
    fing_num = 0
    for finger in forward_dict:
        unit_vector = torch.zeros(cur_tensor.shape[0], cur_tensor.shape[1], 4, 1, device=cur_tensor.device,
                                  dtype=cur_tensor.dtype)
        unit_vector[:,:,3,:] = 1 #whatever the name is for the vector thing with a 1

        unit_vector[:,:,2,:] = fing_num*segment_length/3

        unit_vector[:,:,0,:] = segment_length

        prev_tensor = torch.zeros(cur_tensor.shape[0], cur_tensor.shape[1], 4, 1, device=cur_tensor.device,
                                  dtype=cur_tensor.dtype)
        prev_tensor[:, :, 3, :] = 1  # whatever the name is for the vector thing with a 1

        prev_tensor[:, :, 2, :] = fing_num * segment_length / 3

        new_tensor[:, :, i * 3:i * 3 + 3] = prev_tensor[:, :, 0:3, 0]
        i = i+1
        for finger_angle_i in range(len(finger['finger_angles'])):
            rotation_indices = []
            for finger_angle_i_sum in range(finger_angle_i + 1):
                rotation_indices.append(finger['finger_angles'][finger_angle_i_sum])
            rotation_amount = torch.sum(cur_tensor[:,:,rotation_indices], dim=2)
            rotation_matrix = generate_rotation_matrix(rotation_amount)
            #translation_matrix =generate_translation_matrix(translation_amount)
            #transformation_matrix = torch.matmul(rotation_matrix, translation_matrix)
            prev_tensor = torch.matmul(rotation_matrix, unit_vector) + prev_tensor
            new_tensor[:, :, i * 3:i * 3 + 3] = prev_tensor[:, :, 0:3, 0]
            i = i+1
        fing_num = fing_num + 1
    return new_tensor


def generate_rotation_matrix(rotation_amount):
     new_tensor = torch.zeros(rotation_amount.shape[0], rotation_amount.shape[1], 4,4, device=rotation_amount.device, dtype=rotation_amount.dtype)
     new_tensor[:,:,0,0] = torch.cos(rotation_amount)
     new_tensor[:, :, 0, 1] = -torch.sin(rotation_amount)
     new_tensor[:, :, 1, 0] = torch.sin(rotation_amount)
     new_tensor[:, :, 1, 1] = torch.cos(rotation_amount)
     new_tensor[:, :, 2, 2] = 1
     new_tensor[:, :, 3, 3] = 1
     return new_tensor

def generate_translation_matrix(translation_amount):
    new_tensor = torch.zeros(translation_amount.shape[0], translation_amount.shape[1], 4, 4, device=translation_amount.device,
                             dtype=translation_amount.dtype)
    new_tensor[:, :, 0, 0] = 1
    new_tensor[:, :, 1, 1] = 1
    new_tensor[:, :, 2, 2] = 1
    new_tensor[:, :, 3, 3] = 1
    new_tensor[:, :, 0, 3] = translation_amount
    return new_tensor