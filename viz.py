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
        j = 0
        for hand in hand_type:
            hand3d = hand.reshape((-1, 3)).transpose()

            # Plot edges for each bone
            color_mod =(j+1)/len(hand_type)
            out_color = [1-(1-c)*color_mod for c in colors[i]]
            out_color[3] =1
            for edge in hand_edges:
                ax.plot(hand3d[0, edge], hand3d[1, edge], hand3d[2, edge], color=out_color)
            ax.scatter(hand3d[0, :], hand3d[1, :], hand3d[2, :], color=out_color)
            j +=1

        #
        #
        # # Plot edges for each bone
        # for edge in hand_edges:
        #     ax.plot(hand3d[0, edge], hand3d[1, edge], hand3d[2, edge], color=colors[i])

        i=i+1
    axes = plt.gca()
    # Hide grid lines
    axes.grid(False)

    # Hide axes ticks
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_zticks([])

    # Get rid of the panes
    axes.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axes.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axes.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    axes.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    axes.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    axes.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    plt.show()

def plot_loss(losses, title = "losses", ymax=50):
    for loss in losses:
        plt.plot(range(len(loss)), loss)
    axes = plt.gca()
    #axes.set_xlim([xmin, xmax])
    axes.set_ylim([0, ymax])
    plt.legend([r'$loss(Y, \hat{Y})$',r'$loss(X, \hat{Y})$',r'$loss(X, Y)$'])
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()
def plot_double_loss(losses1, losses2, losses3, title = "losses", ymax=50):
    for loss in losses1:
        plt.plot(range(len(loss)), loss)
    for loss in losses2:
        plt.plot(range(len(loss)), loss)
    for loss in losses3:
        plt.plot(range(len(loss)), loss)
    axes = plt.gca()
    #axes.set_xlim([xmin, xmax])
    axes.set_ylim([0, ymax])
    plt.legend(['angle to reconstructed loss','reconstructed loss', 'baseline'])
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()