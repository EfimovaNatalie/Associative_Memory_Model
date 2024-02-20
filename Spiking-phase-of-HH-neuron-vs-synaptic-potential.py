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
k_t = 0.05#0.2
teta_syn = 0
k_syn = 0.2
tau_m = 0.16
g_syn = 0.1

I_input = 0.1 # input current onto pre oscillator
V_th = 0 # threshold potential
V_0 = -1.2 # starting membrane potential value
w_0 = g_slow*V_0
F_0 = 1 # starting relative phase

V_syn_list = np.linspace(-4, 5, 100)

def I_fast(V):
    return (-V + np.tanh(g_fast*V))
def W_inf(V):
    return g_slow*V
def tau_w(V):
    return tau_2 + (tau_1 - tau_2)/(1 + np.exp(-V/k_t))#/k_t))
def S_inf(V):
    return 1/(1 + np.exp((teta_syn - V)/k_syn))
'''def V_s(x):
    return a*x + b'''
def I_syn(V_i, V_j, s_ij):
    return g_syn*S_inf(V_j)*(V_i - V_s(s_ij))

# time linspace
t_start = 0
t_stop = 1500
dt = 0.1
TIME = np.arange(t_start, t_stop, dt)

# presynaptic oscillator initialization
'''V_pre_list = []
t_pre_spikes = []
V_pre = V_0
w_pre = w_0

# postsynaptic oscillator initialization
V_post_list = []
t_post_spikes = []
V_post = V_0
w_post = w_0

F = F_0
F_list = []

F_list.append(F)
V_pre_list.append(V_pre)
V_post_list.append(V_post)'''

F_avr_list = []

V_pre = V_0
w_pre = w_0
t_pre_spikes = []
V_pre_list = []
V_pre_list.append(V_pre)

for step in range(len(TIME) - 1):
    V_pre_old = V_pre
    w_pre_old = w_pre
    dV_pre = (I_fast(V_pre_old) - I_input - w_pre_old) * dt / tau_m
    # print(V_pre_old, tau_w(V_pre_old), w_pre_old)
    dw_pre = (W_inf(V_pre_old) - w_pre_old) * dt / tau_w(V_pre_old)

    V_pre = V_pre_old + dV_pre
    w_pre = w_pre_old + dw_pre

    if V_pre >= V_th and V_pre_old < V_th:
        t_pre_spikes.append(TIME[step])

    V_pre_list.append(V_pre)

T_pre = t_stop
if len(t_pre_spikes) > 2:
    periods = [t_pre_spikes[i + 1] - t_pre_spikes[i] for i in range(len(t_pre_spikes) - 1)]
    T_pre = sum(periods) / len(periods)

print("T_pre: ", T_pre)

for i in range(len(V_syn_list)):
    V_syn = V_syn_list[i]
    V_post = V_0
    w_post = w_0
    F = F_0

    t_post_spikes = []
    V_post_list = []
    F_list = []

    V_post_list.append(V_post)
    F_list.append(F)

    for step in range(len(TIME) - 1):

        num_post_spikes = 0
        F_old = F

        V_post_old = V_post
        w_post_old = w_post

        dV_post = (I_fast(V_post_old) - g_syn*(V_post_old - V_syn)/(1 + np.exp(-(V_pre_list[step] - teta_syn)/k_syn)) - w_post_old)*dt/tau_m
        # print(V_pre_old, tau_w(V_pre_old), w_pre_old)
        dw_post = (W_inf(V_post_old) - w_post_old) * dt / tau_w(V_post_old)

        V_post = V_post_old + dV_post
        w_post = w_post_old + dw_post

        if V_post >= V_th and V_post_old < V_th:
            t_post_spikes.append(TIME[step])
            t_pre_last = max(filter(lambda x: x <= TIME[step], t_pre_spikes),
                             default=0)  # t_pre_spikes[num_post_spikes]#
            F = (TIME[step] - t_pre_last)/T_pre
            F_list.append(F)
            num_post_spikes += 1
        else:
            F_list.append(F_old)

        V_post_list.append(V_post)

    F_avr = mean(F_list[int(t_stop-500/dt):])
    F_avr_list.append(F_avr)

    T_post = t_stop
    if len(t_post_spikes) > 2:
        periods = [t_post_spikes[i + 1] - t_post_spikes[i] for i in range(20, len(t_post_spikes) - 1)]
        T_post = sum(periods) / len(periods)
        #print("V_syn: ", V_syn)
        #print("T_post: ", T_post, '\n')
    if i == 99:#(0 <= V_syn < 0.1):#(-0.4 < V_syn < -0.2):#(len(V_syn_list)-1):
        print("nu_pre: ", 1 / T_pre)
        print("nu_post: ", 1 / T_post)
        plt.figure(figsize=(8, 8))
        plt.plot(TIME, V_pre_list, label=r'$V_{pre}$', c='red', lw=1)
        plt.plot(TIME, V_post_list, label=r'$V_{post}$', c='green', lw=1)
        plt.legend()
        plt.figure(figsize=(8, 8))
        plt.plot(TIME, F_list, label='phase', c='blue', lw=2)
        plt.legend()


'''print(r"$\nu_{pre}$: ", 1/T_pre)
print(r"$\nu_{post}$: ", 1/T_post)

plt.figure(figsize=(8,8))
plt.plot(TIME, V_pre_list, label=r'$V_{pre}$', c='red', lw=1)
plt.plot(TIME, V_post_list, label=r'$V_{post}$', c='green', lw=1)
plt.legend()

plt.figure(figsize=(8,8))
plt.plot(TIME, F_list_new, label='phase', c='blue', lw=2)
plt.legend()'''

plt.figure(figsize=(8,8))
plt.scatter(V_syn_list, F_avr_list, c='red', s=10)
plt.xlabel(r"$V_{syn}$")
plt.ylabel(r"$\phi$")

plt.show()