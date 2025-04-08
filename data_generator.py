#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

rng = np.random.default_rng()

def venn_concentration(x, nx, c_mix):
    #give concentration value based on x coordinate
    if x < nx/3: 
        return [1, 0]
    elif x < 2*nx/3:
        return [c_mix, 1-c_mix]
    else:
        return [0, 1]
    
def generate_concentrations(frame_shape, c_mix=0.5, venn=False):
    (nx, ny) = frame_shape

    if venn:
        concentrations = np.array([[venn_concentration(x, nx, c_mix) for x in range(nx)] for y in range(ny)]).transpose((1, 0, 2))
    else:
        concentrations = np.array([[venn_concentration(nx/2, nx, c_mix) for x in range(nx)] for y in range(ny)]).transpose((1, 0, 2)) #only mixed

    return concentrations

def generate_measurements(decay_rates, concentrations, n):
    #n: the number of excitations per pixel
    decay_rates=np.array(decay_rates)
    nd = len(decay_rates)

    pixel_dye_indices = np.apply_along_axis(lambda cs: np.random.choice(np.arange(nd), n, p=cs), axis=2, arr=concentrations)

    pixel_decay_rates = decay_rates[pixel_dye_indices]
    
    rng = np.random.default_rng()
    arrival_times = rng.exponential(1/pixel_decay_rates)

    return arrival_times


#%%
#Parameters:
nx, ny = 50, 50
c_mix = 0.4
l1 = 0.05
l2 = 0.09

c = generate_concentrations((nx, ny), c_mix, False)
decay_rates = np.array([l1, l2])
arrival_times = generate_measurements(decay_rates, c, 1000) #photon arrival times

#continuous: np.save('continuous_0.4.npy', arrival_times)

t_bins = np.arange(0, np.ceil(np.max(arrival_times)))#np.round(10/np.min(decay_rates)))
binned_data = np.apply_along_axis(lambda arr: np.histogram(arr, t_bins)[0], axis=2, arr=arrival_times)
np.save('gen_{c_mix}.npy', binned_data) 

#%% Generate data sets with range of concentrations
path = 'GenData/'
nx, ny = 10, 10
c1s = np.linspace(0.01, 0.99, num=20)

for c1 in c1s:

    c = generate_concentrations((nx, ny), c1, False)
    decay_rates = np.array([0.05, 0.09])
    arrival_times = generate_measurements(decay_rates, c, 1000)
    
    t_bins = np.arange(0, np.ceil(np.max(arrival_times)))#np.round(10/np.min(decay_rates)))
    binned_data = np.apply_along_axis(lambda arr: np.histogram(arr, t_bins)[0], axis=2, arr=arrival_times)
    np.save(path+f'gen_{round(c1, 3)}.npy', binned_data)
