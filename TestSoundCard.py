# %% 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


# %% 

# class Measurement:
# class Signal:
#     f_s = 44100                              # Sampling rate (Hz)
    
#     def sine(self):
#         toto = 1
#         return toto
#     def multitone(self):
#         toto = 1
#         return toto
#     def sweep(self):
#         toto = 1        
#         return toto

# %% Single tone --------------------------------
f_s = 44100                                     # Sampling rate (Hz)
T_target = 5                                    # Signal duration (s)
t_target = np.arange(T_target * f_s) / f_s      # Time axis for target duration


f_1 = 500                                        # Sine frequency
T_1 = 1 / f_1                                   # Corresponding period for sine
x = np.sin(2*np.pi * t_target / T_1)            # Full sine signal

N = int(T_target / T_1)                         # Number of periods in signal (sine) to have an integer number of periods T_1 in T
N_s = int(N * T_1 * f_s)                        # Number of sample in trimmed signal
t_sine = np.arange(N_s) / f_s                   # Time axis for trimmed signal
x_sine = x[:N_s]                                # Trimmed sine signal with integer n° of periods


# %% Multi tone --------------------------------


# %% Swept sine --------------------------------
f_0 = 1e2
f_1 = 1e3

SweepType = "lin"
if SweepType == "lin":
    f_m = (f_1 - f_0) / T_target
    f_t = f_m * t_target
    x_sweep = np.sin(2*np.pi * f_t * t_target + f_0)
elif SweepType == "log":
    A = f_0
    alpha = (1/T_target) * np.log(f_1 / f_0)
    f_t = A * np.exp(alpha * t_target)
    x_sweep = np.sin(f_t * t_target)



[fig,ax] = plt.subplots()
ax.plot(t_target,f_t)

# %% Play sound
sd.play(x_sweep,f_s)


# %%
[fig,ax] = plt.subplots(2,1)
ax[0].plot(t_sine,x_sine)
ax[1].plot(t_target,x_sweep)
plt.show()