import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['image.interpolation'] = 'nearest'

def visualize_hands(hand_type_in):
    colors = plt.cm.hsv(np.linspace(0, 1, 1+len(hand_type_in))).tolist()
    print(colors)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-90, azim=-90)
    # ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.axis('equal')

    '''
    ## Visualize 3D Hand
    '''
    # Joint orders follow Openpose keypoint output
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
    hand_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
                           [0, 5], [5, 6], [6, 7], [7, 8],
                           [0, 9], [9, 10], [10, 11], [11, 12],
                           [0, 13], [13, 14], [14, 15], [15, 16],
                           [0, 17], [17, 18], [18, 19], [19, 20]])

    # 3D hands, right_hand and left_hand, have 21 3D joints, stored as an array [x1,y1,z1,x2,y2,z2,...]

    '''
    # Right hand
    '''
    i = 0
    for hand_type in hand_type_in:
        for hand in hand_type:
            #print(hand_array_test.shape)
            #print(hand.shape)
            hand3d = hand.reshape((-1, 3)).transpose()

            # Plot edges for each bone
            for edge in hand_edges:
                ax.plot(hand3d[0, edge], hand3d[1, edge], hand3d[2, edge], color=colors[i])
            ax.scatter(hand3d[0, :], hand3d[1, :], hand3d[2, :], color=colors[i])

        #
        #
        # # Plot edges for each bone
        # for edge in hand_edges:
        #     ax.plot(hand3d[0, edge], hand3d[1, edge], hand3d[2, edge], color=colors[i])

        i=i+1

    plt.show()

def plot_loss(losses):
    for loss in losses:
        plt.plot(range(len(loss)), loss)
    axes = plt.gca()
    #axes.set_xlim([xmin, xmax])
    axes.set_ylim([0, 50])
    plt.legend([r'$loss(Y, \hat{Y})$',r'$loss(X, \hat{Y})$',r'$loss(X, Y)$'])
    plt.title("Test losses")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()