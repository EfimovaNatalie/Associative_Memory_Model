import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import random
from statistics import mean

# initialization of parameters
teta_syn = -5
k_syn = 10
g_syn = 0.7

a = -10
b = 5

I_input = 7 # input current onto base oscillator

# HH neuron parameters
V_rest = -60  # mV, resting membrane potential
V_th = 0  # mV, threshold potential

E_k = -77  # mV, equilibrium potassium potential
E_Na = 50  # mV, equilibrium sodium potential
E_L = -54.4  # mV, equilibrium leakage potential

g_K = 36  # mSm, membrane conductivity for potassium current
g_Na = 120  # mSm, membrane conductivity for sodium current
g_L = 0.3  # mSm, membrane conductivity for leakage current

C = 1  # microF, membrane capacitance

# initial values for gate HH neuron variables
h_0 = 0.6
n_0 = 0.3
m_0 = 0.05

V_0 = V_rest  # initial membrane potential
F_0 = 1  # starting relative phase

# definition of gate variables for Hodgkin Huxley neuron modeling
def a_n(v):
    return (0.01 * v + 0.55) / (1 - np.exp(-0.1 * v - 5.5))
def b_n(v):
    return 0.125 * np.exp(-(v + 65) / 80)
def a_m(v):
    return (0.1 * v + 4) / (1 - np.exp(-0.1 * v - 4))
def b_m(v):
    return 4 * np.exp(-(v + 65) / 18)
def a_h(v):
    return 0.07 * np.exp(-(v + 65) / 20)
def b_h(v):
    return 1 / (1 + np.exp(-0.1 * v - 3.5))
def S_inf(V):
    return 1 / (1 + np.exp((teta_syn - V) / k_syn))
def V_s(x):
    return a * x + b

# time linspace
t_start = 0
t_stop = 200
dt = 0.01
TIME = np.arange(t_start, t_stop, dt)

# initial parameters values for basesynaptic neuron
V_base = V_0
h_base = h_0
n_base = n_0
m_base = m_0
t_base_spikes = []
V_base_list = []
V_base_list.append(V_base)

# base oscillator dynamics
for step in range(len(TIME) - 1):
    V_base_old = V_base

    dV_base = (I_input - g_K *(n_base**4)*(V_base_old - E_k) - g_Na*(m_base**3)*h_base*(V_base_old - E_Na) - g_L*(V_base_old - E_L))*dt/C
    dn_base = (a_n(V_base_old) * (1 - n_base) - b_n(V_base_old) * n_base) * dt
    dm_base = (a_m(V_base_old) * (1 - m_base) - b_m(V_base_old) * m_base) * dt
    dh_base = (a_h(V_base_old) * (1 - h_base) - b_h(V_base_old) * h_base) * dt
    V_base = V_base_old + dV_base
    n_base += dn_base
    m_base += dm_base
    h_base += dh_base

    if V_base >= V_th and V_base_old < V_th:
        t_base_spikes.append(TIME[step])

    V_base_list.append(V_base)

#print("V_base: ", V_base_list)
#plt.plot(TIME, V_base_list, label='V_base')
#plt.legend()

T_base = t_stop # base oscillations' period
if len(t_base_spikes) > 2:
    periods = [t_base_spikes[i + 1] - t_base_spikes[i] for i in range(len(t_base_spikes) - 1)]
    T_base = sum(periods) / len(periods)

N = 60  # number of units in a single layer

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

input_image_zero = [[-1, -1, -1, 1, -1, 1], # "broken" image "0"
                       [1, -1, -1, -1, -1, 1],
                       [-1, -1, 1, 1, -1, -1],
                       [-1, 1, -1, 1, -1, -1],
                       [-1, 1, 1, 1, 1, -1],
                       [-1, 1, 1, -1, 1, 1],
                       [-1, 1, 1, 1, -1, -1],
                       [-1, -1, 1, 1, -1, -1],
                       [1, -1, -1, -1, -1, 1],
                       [1, -1, -1, -1, -1, 1]]

input_image_one = [[-1, 1, -1, -1, 1, 1],  # "broken" image "1" to be retrieved
                   [1, -1, -1, -1, -1, 1],
                   [1, -1, -1, -1, 1, -1],
                   [1, 1, 1, -1, 1, 1],
                   [1, 1, -1, -1, 1, 1],
                   [1, 1, -1, 1, 1, 1],
                   [1, 1, -1, -1, 1, 1],
                   [1, 1, 1, -1, 1, 1],
                   [1, 1, -1, -1, 1, 1],
                   [-1, 1, -1, -1, -1, -1]]

input_image_two = [[1, -1, -1, -1, -1, 1],  # encodes image "2"
                         [-1, -1, -1, -1, -1, -1],
                         [-1, 1, 1, 1, -1, -1],
                         [-1, 1, 1, 1, -1, 1],
                         [1, 1, 1, 1, -1, -1],
                         [1, 1, 1, -1, -1, 1],
                         [1, 1, -1, -1, 1, 1],
                         [1, -1, -1, 1, 1, 1],
                         [1, 1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1, 1]]

# convert matrices of images to vectors
memorized_1 = np.array(memorized_pattern_zero).reshape(1, 60)
memorized_2 = np.array(memorized_pattern_one).reshape(1, 60)
memorized_3 = np.array(memorized_pattern_two).reshape(1, 60)
input_img = np.array(input_image_two).reshape(1, 60)

# print(input_img.shape)

# set of vectors to be memorized
memorized = [memorized_1[0], memorized_2[0], memorized_3[0]]

# initialization of "Connectivity" matrix taking into account Hebbian learning rule
S = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        s_ij = 0
        for k in range(len(memorized)):
            s_ij += memorized[k][i] * memorized[k][j]
        S[i][j] += s_ij#/N

# initialization of input layer units
V_inp_units = V_0 * np.ones(N)
n_inp_units = n_0 * np.ones(N)
m_inp_units = m_0 * np.ones(N)
h_inp_units = h_0 * np.ones(N)
V_inp_units_list = []
F_inp_units = F_0 * np.ones(N)

# synaptic potentials between base oscillator and input layer initialization
V_syn_list = V_s(input_img[0])

# Control layer initialization
V_out_units = V_0 * np.ones(N)
n_out_units = n_0 * np.ones(N)
m_out_units = m_0 * np.ones(N)
h_out_units = h_0 * np.ones(N)
F_out_units = F_0 * np.ones(N)

V_post = []
num_of_spikes_list = np.zeros(N)

for step in range(len(TIME)):
    print("%: ", 100*step/len(TIME))

    # input layer modeling
    V_inp_units_old = V_inp_units.copy()
    V_inp_units_list.append(V_inp_units_old)

    #print("V_inp_units_old: ", V_inp_units_old)

    for i in range(N):
        F_inp = F_inp_units[i]
        V_old = V_inp_units_old[i]
        n_old = n_inp_units[i]
        m_old = m_inp_units[i]
        h_old = h_inp_units[i]

        V_syn = V_syn_list[i]

        I_syn = g_syn * (V_old - V_syn) / (1 + np.exp(-(V_base_list[step] - teta_syn) / k_syn))
        sign = 0
        if V_syn > 0:
            sign = 1
        else:
            sign = -1
        #print("I_syn: ", sign*I_syn)
        dV = (I_input - sign*I_syn - g_K*(n_old**4)*(V_old - E_k) - g_Na * (m_old**3)*h_old*(V_old - E_Na) - g_L*(V_old - E_L))*dt/C
        #print(dV)
        dn = (a_n(V_old) * (1 - n_old) - b_n(V_old) * n_old) * dt
        dm = (a_m(V_old) * (1 - m_old) - b_m(V_old) * m_old) * dt
        dh = (a_h(V_old) * (1 - h_old) - b_h(V_old) * h_old) * dt

        V = V_old + dV
        n_old += dn
        m_old += dm
        h_old += dh

        V_inp_units[i] = V
        #print(V_inp_units[i], V_inp_units_old[i], dV)
        n_inp_units[i] = n_old
        m_inp_units[i] = m_old
        h_inp_units[i] = h_old

        if V >= V_th and V_old < V_th:
            t_pre_last = max(filter(lambda x: x <= TIME[step], t_base_spikes), default=0)
            F_inp = (TIME[step] - t_pre_last) / T_base
            F_inp_units[i] = F_inp
        else:
            F_inp_units[i] = F_inp
    #print(V_inp_units - V_inp_units_old)
    V_out_units_old = V_out_units.copy()

    for i in range(N):

        F_out = F_out_units[i]
        V_old = V_out_units_old[i]
        n_old = n_out_units[i]
        m_old = m_out_units[i]
        h_old = h_out_units[i]

        I_out_syn = 0
        for j in range(N):
            V_syn = V_s(S[i][j])
            sign = 0
            if V_syn > 0:
                sign = 1
            else:
                sign = -1
            I_out_syn += sign*g_syn*(V_old - V_syn)/(N*(1 + np.exp((teta_syn - V_inp_units_old[j])/k_syn)))
        dV = (I_input + I_out_syn - g_K*(n_old**4)*(V_old - E_k) - g_Na*(m_old**3)*h_old*(V_old - E_Na) - g_L*(V_old - E_L))*dt/C
        dn = (a_n(V_old) * (1 - n_old) - b_n(V_old) * n_old) * dt
        dm = (a_m(V_old) * (1 - m_old) - b_m(V_old) * m_old) * dt
        dh = (a_h(V_old) * (1 - h_old) - b_h(V_old) * h_old) * dt

        V = V_old + dV
        n_old += dn
        m_old += dm
        h_old += dh

        if (i == 5):
            V_post.append(V)

        V_out_units[i] = V
        n_out_units[i] = n_old
        m_out_units[i] = m_old
        h_out_units[i] = h_old

        if V >= V_th and V_old < V_th:
            num_of_spikes_list[i] += 1
            t_pre_last = max(filter(lambda x: x <= TIME[step], t_base_spikes), default=0)
            F_out = (TIME[step] - t_pre_last) / T_base
            F_out_units[i] = F_out
        else:
            F_out_units[i] = F_out

input_layer = np.array(F_inp_units * np.pi).reshape(10, 6)
retrieved_result = np.array(F_out_units*np.pi).reshape(10, 6)

plt.figure(figsize=(6,3))
plt.plot(TIME, V_post, label='V_post')
plt.plot(TIME, V_base_list, label='V_base')
plt.legend()

#print(F_inp_units)
V = [V_inp_units_list[x][1] for x in range(len(TIME))]

'''plt.plot(TIME, V, label="V_input_unit")
plt.plot(TIME, V_base_list, label="V_base")
plt.legend()'''

fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
ax1, ax2 = axes

im1 = ax1.matshow(np.cos(input_layer))
im2 = ax2.matshow(np.cos(retrieved_result))
ax1.set_title("Input image")
ax2.set_title('Retrieved result, Time = %i' %t_stop)
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

neuron_number = [x for x in range(N)]

plt.figure()
plt.plot(neuron_number, num_of_spikes_list)

plt.show()
