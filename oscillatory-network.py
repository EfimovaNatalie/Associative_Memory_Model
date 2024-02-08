'''Two-layer phase oscillatory network is modeled
 for retrieving dynamics demonstration'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import random

N = 60 # number of units in a single layer

memorized_pattern_zero = [[1, -1, -1, -1, -1, 1], # encodes image "0"
                       [1, -1, -1, -1, -1, 1],
                       [-1, -1, 1, 1, -1, -1],
                       [-1, -1, 1, 1, -1, -1],
                       [-1, 1, 1, 1, 1, -1],
                       [-1, 1, 1, 1, 1, -1],
                       [-1, -1, 1, 1, -1, -1],
                       [-1, -1, 1, 1, -1, -1],
                       [1, -1, -1, -1, -1, 1],
                       [1, -1, -1, -1, -1, 1]]
memorized_pattern_one = [[1, 1, -1, -1, 1, 1], # encodes image "-1"
                       [1, -1, -1, -1, 1, 1],
                       [-1, -1, -1, -1, 1, 1],
                       [1, 1, -1, -1, 1, 1],
                       [1, 1, -1, -1, 1, 1],
                       [1, 1, -1, -1, 1, 1],
                       [1, 1, -1, -1, 1, 1],
                       [1, 1, -1, -1, 1, 1],
                       [1, 1, -1, -1, 1, 1],
                       [-1, -1, -1, -1, -1, -1]]
memorized_pattern_two = [[1, -1, -1, -1, -1, 1], # encodes image "1"
                       [-1, -1, -1, -1, -1, -1],
                       [-1, -1, 1, 1, -1, -1],
                       [1, 1, 1, 1, -1, -1],
                       [1, 1, 1, 1, -1, -1],
                       [1, 1, 1, -1, -1, 1],
                       [1, 1, -1, -1, 1, 1],
                       [1, -1, -1, 1, 1, 1],
                       [-1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1]]

input_image_zero = [[1, -1, 1, -1, 1, 1], # "broken" image "0"
                       [1, -1, -1, -1, -1, 1],
                       [1, -1, 1, 1, -1, -1],
                       [-1, 1, -1, -1, 1, -1],
                       [-1, 1, 1, -1, 1, -1],
                       [-1, 1, -1, 1, -1, -1],
                       [-1, 1, 1, 1, -1, 1],
                       [-1, -1, 1, 1, -1, -1],
                       [1, -1, -1, -1, -1, 1],
                       [1, -1, 1, -1, -1, 1]]

input_image_one = [[-1, 1, -1, -1, 1, 1], # "broken" image "1" to be retrieved
               [1, -1, -1, -1, -1, 1],
               [1, -1, -1, -1, 1, -1],
               [1, 1, 1, -1, 1, 1],
               [1, 1, -1, -1, 1, 1],
               [1, 1, -1, 1, 1, 1],
               [1, 1, -1, -1, 1, 1],
               [1, 1, 1, -1, 1, 1],
               [1, 1, -1, -1, 1, 1],
               [-1, 1, -1, -1, -1, -1]]
input_image_two = [[1, -1, 1, -1, -1, 1], # "broken" image "2" to be retrieved
                   [1, -1, -1, -1, 1, -1],
                   [-1, -1, 1, -1, -1, -1],
                   [1, -1, 1, 1, -1, 1],
                   [1, 1, 1, -1, -1, -1],
                   [1, 1, 1, -1, -1, 1],
                   [1, 1, -1, -1, 1, 1],
                   [-1, -1, -1, 1, 1, 1],
                   [1, -1, -1, -1, -1, -1],
                   [-1, -1, -1, -1, 1, -1]]

# convert matrices of images to vectors
memorized_1 = np.array(memorized_pattern_zero).reshape(1, 60)
memorized_2 = np.array(memorized_pattern_one).reshape(1, 60)
memorized_3 = np.array(memorized_pattern_two).reshape(1, 60)
input_img = np.array(input_image_one).reshape(1, 60)

#print(input_img.shape)

# set of vectors to be memorized
memorized = [memorized_1[0], memorized_2[0], memorized_3[0]]

# initialization of "Connectivity" matrix taking into account Hebbian learning rule
S = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        s_ij = 0
        for k in range(len(memorized)):
            s_ij += memorized[k][i]*memorized[k][j]
        S[i][j] += s_ij/N

# time linspace
t_start = 0
t_stop = 10
dt = 1
TIME = np.arange(t_start, t_stop, dt)

#control layer initialization (random phases for a start)
control_layer = np.array([random.random() for i in range(60)])

#input layer initialization from input image
input_layer = np.array([np.arccos(xi) for xi in input_img[0]])

for step in range(len(TIME)):
    for i in range(N): # loop along the control layer units
        phi_i_old = control_layer[i]
        dphi_i = 0
        for j in range(N): # loop along the input layer units
            phi_j = input_layer[j]
            dphi_ij = S[i][j]*np.sin(phi_j - phi_i_old)*dt
            dphi_i += dphi_ij
        phi_i = phi_i_old + dphi_i
        control_layer[i] = phi_i

input_layer = np.array(input_img).reshape(10, 6)
result_in_control_layer = [np.cos(phi) for phi in control_layer]
retrieved_result = np.array(result_in_control_layer).reshape(10, 6)

fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
ax1, ax2 = axes

im1 = ax1.matshow(input_layer)
im2 = ax2.matshow(retrieved_result)
ax1.set_title("Input image")
ax2.set_title('Retrieved result, Time = %i' %t_stop)
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

plt.show()

