import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import random
from statistics import mean

# initialization of parameters
g_fast = 2
g_slow = 2
tau_1 = 5
tau_2 = 50
k_t = 0.05
teta_syn = 0
k_syn = 0.2
tau_m = 0.16
g_syn = 0.1

I_input = 0.1  # input current onto base oscillator
V_th = 0  # threshold potential
V_0 = -1.2  # starting membrane potential value
w_0 = g_slow * V_0
F_0 = 1  # starting relative phase (in radians)
N = 60  # number of units in a single layer

a = -3
b = -1

def I_fast(V):
    return (-V + np.tanh(g_fast * V))
def W_inf(V):
    return g_slow * V
def tau_w(V):
    return tau_2 + (tau_1 - tau_2) / (1 + np.exp(-V / k_t))  # /k_t))
def S_inf(V):
    return 1 / (1 + np.exp((teta_syn - V) / k_syn))
def V_s(x):
    return a * x + b
def I_syn(V_i, V_j, s_ij):
    return g_syn * S_inf(V_j) * (V_i - V_s(s_ij))

# time linspace
t_start = 0
t_stop = 400
dt = 0.1
TIME = np.arange(t_start, t_stop, dt)

memorized_pattern_zero = [[1, -1, -1, -1, -1, 1],  # encodes image "0"
                          [1, -1, -1, -1, -1, 1],
                          [-1, -1, 1, 1, -1, -1],
                          [-1, -1, 1, 1, -1, -1],
                          [-1, 1, 1, 1, 1, -1],
                          [-1, 1, 1, 1, 1, -1],
                          [-1, -1, 1, 1, -1, -1],
                          [-1, -1, 1, 1, -1, -1],
                          [1, -1, -1, -1, -1, 1],
                          [1, -1, -1, -1, -1, 1]]

memorized_pattern_one = [[1, 1, -1, -1, 1, 1],  # encodes image "-1"
                         [1, -1, -1, -1, 1, 1],
                         [-1, -1, -1, -1, 1, 1],
                         [1, 1, -1, -1, 1, 1],
                         [1, 1, -1, -1, 1, 1],
                         [1, 1, -1, -1, 1, 1],
                         [1, 1, -1, -1, 1, 1],
                         [1, 1, -1, -1, 1, 1],
                         [1, 1, -1, -1, 1, 1],
                         [-1, -1, -1, -1, -1, -1]]

memorized_pattern_two = [[1, -1, -1, -1, -1, 1],  # encodes image "1"
                         [-1, -1, -1, -1, -1, -1],
                         [-1, -1, 1, 1, -1, -1],
                         [1, 1, 1, 1, -1, -1],
                         [1, 1, 1, 1, -1, -1],
                         [1, 1, 1, -1, -1, 1],
                         [1, 1, -1, -1, 1, 1],
                         [1, -1, -1, 1, 1, 1],
                         [-1, -1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, -1]]

input_image_zero = [[1, -1, 1, -1, 1, 1],  # "broken" image "0"
                    [1, -1, -1, -1, -1, 1],
                    [1, -1, 1, 1, -1, -1],
                    [-1, 1, -1, -1, 1, -1],
                    [-1, 1, 1, -1, 1, -1],
                    [-1, 1, -1, 1, -1, -1],
                    [-1, 1, 1, 1, -1, 1],
                    [-1, -1, 1, 1, -1, -1],
                    [1, -1, -1, -1, -1, 1],
                    [1, -1, 1, -1, -1, 1]]

input_image_one = [[1, 1, -1, -1, 1, 1],  # "broken" image "1" to be retrieved
                   [1, -1, -1, -1, -1, 1],
                   [1, -1, -1, -1, 1, 1],
                   [1, 1, 1, -1, 1, 1],
                   [1, 1, -1, -1, 1, 1],
                   [1, 1, -1, 1, 1, 1],
                   [1, 1, -1, -1, 1, 1],
                   [1, 1, -1, -1, 1, 1],
                   [1, 1, -1, -1, 1, 1],
                   [-1, -1, -1, -1, -1, -1]]

input_image_two = [[1, -1, 1, -1, -1, 1],  # "broken" image "2" to be retrieved
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
input_img = np.array(input_image_zero).reshape(1, 60)

# set of vectors to be memorized
memorized = [memorized_1[0], memorized_2[0], memorized_3[0]]

# initialization of "Connectivity" matrix taking into account Hebbian learning rule
S = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        s_ij = 0
        for k in range(len(memorized)):
            s_ij += memorized[k][i] * memorized[k][j]
        S[i][j] = s_ij#/N

'''for i in S:
    for j in i:
        print(V_s(j))'''

# base oscillator modeling
V_base = V_0
w_base = w_0
t_base_spikes = []
V_base_list = []
V_base_list.append(V_base)

for step in range(len(TIME) - 1):
    V_base_old = V_base
    w_base_old = w_base
    dV_base = (I_fast(V_base_old) - I_input - w_base_old) * dt / tau_m
    # print(V_base_old, tau_w(V_base_old), w_base_old)
    dw_base = (W_inf(V_base_old) - w_base_old) * dt / tau_w(V_base_old)

    V_base = V_base_old + dV_base
    w_base = w_base_old + dw_base

    if V_base >= V_th and V_base_old < V_th:
        t_base_spikes.append(TIME[step])

    V_base_list.append(V_base)

T_base = t_stop
if len(t_base_spikes) > 2:
    periods = [t_base_spikes[i + 1] - t_base_spikes[i] for i in range(len(t_base_spikes) - 1)]
    T_base = sum(periods) / len(periods)

# initialization of input layer units
V_inp_units = V_0 * np.ones(N)
V_inp_units_list = []
w_inp_units = w_0 * np.ones(N)
F_inp_units = F_0 * np.ones(N)

# synaptic potentials between base oscillator and input layer initialization
V_syn = V_s(input_img[0])

#print(V_syn.reshape(10, 6), '\n')
#print(V_s(S))

# Control layer initialization
V_out_units = V_0 * np.ones(N)
F_out_units = F_0 * np.ones(N)
w_out_units = w_0 * np.ones(N)

V_post = []
t_post_spikes = []

F_list = []

for step in range(len(TIME)):
    #print("%: ", 100*step/len(TIME))
    V_inp_units_old = V_inp_units.copy()
    w_inp_units_old = w_inp_units.copy()

    V_inp_units_list.append(V_inp_units_old)

    for i in range(N):
        F_inp = F_inp_units[i]
        V_inp_unit_old = V_inp_units_old[i]
        w_inp_unit_old = w_inp_units_old[i]
        dV_inp_unit = (I_fast(V_inp_unit_old) - g_syn*S_inf(V_base_list[step])*(V_inp_unit_old - V_syn[i]) - w_inp_unit_old)*dt/tau_m
        dw_inp_unit = (W_inf(V_inp_unit_old) - w_inp_unit_old) * dt / tau_w(V_inp_unit_old)

        V_inp_unit = V_inp_unit_old + dV_inp_unit
        w_inp_unit = w_inp_unit_old + dw_inp_unit

        V_inp_units[i] = V_inp_unit
        w_inp_units[i] = w_inp_unit

        if V_inp_unit >= V_th and V_inp_unit_old < V_th:
            t_pre_last = max(filter(lambda x: x <= TIME[step], t_base_spikes), default=0)
            F_inp = (TIME[step] - t_pre_last)/T_base
            F_inp_units[i] = F_inp  # *np.pi
        else:
            F_inp_units[i] = F_inp

    V_out_units_old = V_out_units.copy()
    w_out_units_old = w_out_units.copy()

    for i in range(N):
        F_out = F_out_units[i]
        V_out_unit_old = V_out_units_old[i]
        w_out_unit_old = w_out_units_old[i]

        if (i == 1):
            V_post.append(V_out_unit_old)

        I_out_syn = 0
        for j in range(N):
            I_out_syn += g_syn * (V_out_unit_old - V_s(S[i][j]))/(N*(1 + np.exp((teta_syn - V_inp_units_old[j])/k_syn))) #* S_inf(V_inp_units_old[j]) / N
        #print("I_out_syn: ", I_out_syn)
        dV_out_unit = (I_fast(V_out_unit_old) + I_out_syn - w_out_unit_old) * dt / tau_m
        dw_out_unit = (W_inf(V_out_unit_old) - w_out_unit_old) * dt / tau_w(V_out_unit_old)

        V_out_unit = V_out_unit_old + dV_out_unit
        w_out_unit = w_out_unit_old + dw_out_unit

        V_out_units[i] = V_out_unit
        w_out_units[i] = w_out_unit

        if V_out_unit >= V_th and V_out_unit_old < V_th:
            if (i == 1):
                t_post_spikes.append(TIME[step])
            t_pre_last = max(filter(lambda x: x <= TIME[step], t_base_spikes), default=0)
            F_out = (TIME[step] - t_pre_last) / T_base
            F_out_units[i] = F_out
        else:
            F_out_units[i] = F_out
        if (i == 1):
            F_list.append(F_out)

T_post = t_stop
if len(t_post_spikes) > 2:
    periods = [t_post_spikes[i + 1] - t_post_spikes[i] for i in range(len(t_post_spikes) - 1)]
    T_post = sum(periods) / len(periods)

print("T_base: ", T_base)
print("T_post: ", T_post)

input_layer = np.array(F_inp_units*np.pi).reshape(10, 6)
retrieved_result = np.array(F_out_units*np.pi).reshape(10, 6)

'''plt.figure(figsize=(8, 4))
plt.plot(TIME, V_post, label='post')
plt.plot(TIME, V_base_list, label='base')
plt.legend()
plt.figure()
plt.plot(TIME, F_list, label='phase')
plt.legend()
'''

fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
ax1, ax2 = axes

im1 = ax1.matshow(np.cos(input_layer))
im2 = ax2.matshow(np.cos(retrieved_result))
ax1.set_title("Input image")
ax2.set_title('Retrieved result, Time = %i' %t_stop)
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

plt.show()