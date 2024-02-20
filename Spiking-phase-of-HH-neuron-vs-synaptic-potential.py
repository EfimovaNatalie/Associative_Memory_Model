import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import random
from statistics import mean

# initialization of parameters
teta_syn = -10
k_syn = 10
g_syn = 0.2

I_input = 7 # input current onto pre oscillator

# HH neuron parameters
V_rest = -60 # mV, resting membrane potential
V_th = 0 # mV, threshold potential

E_k = -77 # mV, equilibrium potassium potential
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

V_0 = V_rest # initial membrane potential

F_0 = 1 # starting relative phase

# definition of gate variables for Hodgkin Huxley neuron modeling
def a_n(v):
    return (0.01*v + 0.55)/(1 - np.exp(-0.1*v - 5.5))
def b_n(v):
    return 0.125*np.exp(-(v + 65)/80)
def a_m(v):
    return (0.1*v + 4)/(1 - np.exp(-0.1*v - 4))
def b_m(v):
    return 4 * np.exp(-(v + 65)/ 18)
def a_h(v):
    return 0.07*np.exp(-(v + 65)/20)
def b_h(v):
    return 1/(1 + np.exp(-0.1*v - 3.5))
'''def I_syn(V_i, V_j, s_ij):
    return g_syn*S_inf(V_j)*(V_i - V_s(s_ij))'''

# time linspace
t_start = 0
t_stop = 500
dt = 0.01
TIME = np.arange(t_start, t_stop, dt)

# initial parameters values for presynaptic neuron
V_pre = V_0
h_pre = h_0
n_pre = n_0
m_pre = m_0
t_pre_spikes = []
V_pre_list = []
V_pre_list.append(V_pre)

for step in range(len(TIME) - 1):
    V_pre_old = V_pre

    dV_pre = (I_input - g_K*(n_pre**4)*(V_pre_old - E_k) - g_Na*(m_pre**3)*h_pre*(V_pre_old - E_Na) - g_L*(V_pre_old - E_L))*dt/C
    dn_pre = (a_n(V_pre_old) * (1 - n_pre) - b_n(V_pre_old) * n_pre) * dt
    dm_pre = (a_m(V_pre_old) * (1 - m_pre) - b_m(V_pre_old) * m_pre) * dt
    dh_pre = (a_h(V_pre_old) * (1 - h_pre) - b_h(V_pre_old) * h_pre) * dt
    V_pre = V_pre_old + dV_pre
    n_pre += dn_pre
    m_pre += dm_pre
    h_pre += dh_pre

    if V_pre >= V_th and V_pre_old < V_th:
        t_pre_spikes.append(TIME[step])

    V_pre_list.append(V_pre)

T_pre = t_stop
if len(t_pre_spikes) > 2:
    periods = [t_pre_spikes[i + 1] - t_pre_spikes[i] for i in range(len(t_pre_spikes) - 1)]
    T_pre = sum(periods)/len(periods)

V_syn_list = np.linspace(-25, 50, 100)

F_avr_list = []

for V_syn in V_syn_list:
    t_post_spikes = []
    V_post_list = []
    F_list = []

    # initial parameters values for postsynaptic neuron
    V_post = V_0
    h_post = h_0
    n_post = n_0
    m_post = m_0
    F = F_0

    V_post_list.append(V_post)
    F_list.append(F)

    #V_syn = 20  # 160

    for step in range(len(TIME) - 1):
        num_post_spikes = 0
        F_old = F
        V_post_old = V_post

        I_syn = g_syn * (V_post_old - V_syn) / (1 + np.exp(-(V_pre_list[step] - teta_syn) / k_syn))

        sign = 0
        if V_syn > 0:
            sign = 1
        else:
            sign = -1
        # changes in postsynaptic neuron
        dV_post = (sign * I_syn - g_K * (n_post ** 4) * (V_post_old - E_k) - g_Na * (m_post ** 3) * h_post * (
                    V_post_old - E_Na) - g_L * (V_post_old - E_L)) * dt / C
        dn_post = (a_n(V_post_old) * (1 - n_post) - b_n(V_post_old) * n_post) * dt
        dm_post = (a_m(V_post_old) * (1 - m_post) - b_m(V_post_old) * m_post) * dt
        dh_post = (a_h(V_post_old) * (1 - h_post) - b_h(V_post_old) * h_post) * dt
        V_post = V_post_old + dV_post
        n_post += dn_post
        m_post += dm_post
        h_post += dh_post
        V_post_list.append(V_post)

        if V_post >= V_th and V_post_old < V_th:
            t_post_spikes.append(TIME[step])
            t_pre_last = max(filter(lambda x: x <= TIME[step], t_pre_spikes),
                             default=0)  # t_pre_spikes[num_post_spikes]#
            F = (TIME[step] - t_pre_last) / T_pre
            F_list.append(F)
            num_post_spikes += 1
        else:
            F_list.append(F_old)

    F_avr = mean(F_list[int(t_stop*0.8/dt):])
    F_avr_list.append(F_avr)

plt.figure(figsize=(8,8))
plt.scatter(V_syn_list, F_avr_list, c='red', s=10)
plt.xlabel(r"$V_{syn}$")
plt.ylabel(r"$\phi$")

plt.show()