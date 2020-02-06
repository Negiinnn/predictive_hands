import numpy as np
from viz import *

def align_hand_dict(hand_dict):
    for hand_id in hand_dict.keys():
        aligned_hands = []
        real_hands = []
        for frame_number in hand_dict[hand_id]:
            for left_right in hand_dict[hand_id][frame_number]:
                landmarks = hand_dict[hand_id][frame_number][left_right]['landmarks']
                hand_dict[hand_id][frame_number][left_right]['aligned_landmarks'] = align_hand(landmarks)

def align_hand(landmarks):
    points = np.copy(np.reshape(landmarks, (-1,3)))
    p1 = points[0,:]
    p2 = points[9,:]
    p3 = points[13,:]
    v1 = p3 - p2
    v2 = p2 - p1
    normal_vector = np.cross(v1,v2)
    normal_vector = normal_vector/np.linalg.norm(normal_vector)

    wrist_vector = v2/np.linalg.norm(v2)

    other_vector = np.cross(wrist_vector, normal_vector)
    other_vector = other_vector/np.linalg.norm(other_vector)

    orthogonal_matrix = np.array([other_vector, wrist_vector,normal_vector])
    rotation_matrix = np.transpose(orthogonal_matrix)

    points = points - points[0,:]
    points = np.matmul(points, rotation_matrix)
    return points.flatten()

