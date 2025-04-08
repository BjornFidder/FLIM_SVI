#%%
import numpy as np
import matplotlib.pyplot as plt

import os

def exp(t, labda):
    return labda * np.exp(-labda*t)

def new_concentration_EM(c, e1, e2, n):
    return 1/n * np.sum(c*e1 / (c*e1 + (1-c)*e2))

def new_lifetimes_EM(c, e1, e2, X):
    P = c * e1 / (c * e1 + (1-c) * e2)
    p1, p2 = (np.sum(P), np.sum(1-P))
    q1, q2 = (np.sum(X*P), np.sum(X*(1-P)))
    return np.array([p1, q1, p2, q2])

def f2X(f, scale):
    f = np.round(f / scale).astype(int)
    X = []
    for i in range(len(f)):
        for j in range(f[i]):
            X.append(i + 0.5)
    return np.array(X)

def hv_to_rgb(hue, value):
    if (value != 0):
        value = value
    X = value * (1 - np.abs(((2 * hue) % 2) - 1))
    if (hue <= 0.5):
        return (value, X, 0)
    else:
        return (X, value, 0)
    
#%% Load data
f = np.load('dataset.npy')
#Or load generated data

t_start = np.argmax(np.sum(f, axis=(0,1)))
f = f[:, :, t_start:]
f_tot = np.sum(f, axis=(0,1))

#Known decay rate values for dataset:
l1 = 0.0323  
l2 = 0.0714

#%% Decay rates & concentration on total intensity
f_scale = 100
X_total = f2X(f_tot, f_scale)

#For continuous data
# X = np.load('continuous_0.4.npy')
# X_total = np.ndarray.flatten(X)

iters = 500
l1s = np.zeros(iters+1)
l2s = np.zeros(iters+1)
cs_tot = np.zeros(iters+1)

l1s[0] = l1
l2s[0] = l2
cs_tot[0] = 0.5

for i in range(iters):
    e1 = exp(X_total, l1s[i])
    e2 = exp(X_total, l2s[i])
    (p1, q1, p2, q2) = new_lifetimes_EM(cs_tot[i], e1, e2, X_total)
    (l1s[i+1], l2s[i+1]) = (p1/q1, p2/q2)
    cs_tot[i+1] = new_concentration_EM(cs_tot[i], e1, e2, len(X_total))

plt.plot(10*l1s, label=f'$\lambda_1$')
plt.plot(10*l2s, label=f'$\lambda_2$')
#plt.plot(cs_tot, label= '$c$')
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Decay rate ($\mathrm{ns}^{-1}$)', fontsize=14)
plt.hlines(0.5, 0, iters, colors='C0', linestyles='dashed')
plt.hlines(0.9, 0, iters, colors='C1', linestyles='dashed')
plt.legend()

#%% Pixel-wise concentration with given decay rates

nx, ny, n_time = np.shape(f)

# For continuous data:
# X = np.load('continuous_0.4.npy')
# nx, ny, _ = np.shape(X)

iters = 50
tol = 0.002

cs = np.zeros((nx, ny, iters+1))

for jx in range(nx):
    for jy in range(ny):

        
        f_pix = f[jx, jy, :]

        X_pix = f2X(f_pix, 1)

        #For continuous data:
        #X_pix = X[jx, jy]

        n = len(X_pix)
        e1 = exp(X_pix, l1)
        e2 = exp(X_pix, l2)

        cs[jx, jy, 0] = 0.5 #c0(l1, l2, f)

        if (n == 0):
            cs[jx, jy, :] = np.full(iters+1, 0.5)

        else:
            
            for i in range(iters):
                cs[jx, jy, i+1] = new_concentration_EM(cs[jx, jy, i], e1, e2, n)
                
                if np.abs(cs[jx, jy, i+1] - cs[jx, jy, i]) < tol:
                    cs[jx, jy, i+1:] = np.full(iters - i, cs[jx, jy, i+1]) 
                    break

    if (jx % 10 == 0 or jx == nx-1):
        print(f"{jx} / {nx-1}")

#%% Full MLE

nx, ny, n_time = np.shape(f)

#Continuous data:
# X = np.load('continuous_0.4.npy')
# nx, ny, _ = np.shape(X)
# nx, ny = (50, 50)

iters = 50

cs = np.zeros((nx, ny, iters+1))
l1s = np.zeros(iters+1)
l2s = np.zeros(iters+1)

cs = np.full((nx, ny,iters+1), 0.5)
l1s[0] = l1
l2s[0] = l2

for i in range(iters):

    lifetime_vars = np.zeros(4, np.float64)

    for jx in range(nx):
        for jy in range(ny):

            f_pix = f[jx, jy, :]

            X_pix = f2X(f_pix, 1)

            #Continuous data: X_pix = X[jx, jy]

            n = len(X_pix)
            e1 = exp(X_pix, l1s[i])
            e2 = exp(X_pix, l2s[i])

            if (n > 0):
            
                    cs[jx, jy, i+1] = new_concentration_EM(cs[jx, jy, i], e1, e2, n)
                    lifetime_vars += new_lifetimes_EM(cs[jx, jy, i], e1, e2, X_pix) * n
        
    p1, q1, p2, q2 = lifetime_vars

    l1s[i+1] = p1 / q1
    l2s[i+1] = p2 / q2

    print(f"Iteration: {i}/{iters}")

#%% Plot decay rates
plt.figure()
plt.plot(10*l1s, label='$\lambda_1$')
plt.plot(10*l2s, label='$\lambda_2$')
plt.hlines(0.5, 0, iters, colors='C0', linestyles='dashed')
plt.hlines(0.9, 0, iters, colors='C1', linestyles='dashed')
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Decay rate ($\mathrm{ns}^{-1}$)', fontsize=14)
plt.legend()

#%% Color mesh concentrations

plt.pcolormesh(cs[iters].transpose())

#%% Color image concentrations
f_aggr = np.sum(f, axis=2)
c = cs[:, :, iters]

colorimage = np.zeros((nx, ny, 3))
for xp in range(nx):
    for yp in range(ny):
        colorimage[xp,yp,:] = hv_to_rgb(c[xp,yp], f_aggr[xp,yp] / np.max(f_aggr))

colorimage = np.flip(np.transpose(colorimage, axes=(1, 0, 2)), axis = 0)
plt.axis('off')
plt.imshow(colorimage)
plt.show()

#%% Concentration comparison on synthetic data

#Load directory containing set of synthetic data sets with varying concentrations
#file names: "gen_c.npy" with c the concentration
directory = os.fsencode('GenData/')

n_c = len(os.listdir(directory))
cs_true = []
cs_avg = []
cs_std = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    c_true = float(filename[4:-4])
    print(c_true)
    l1 = 0.05
    l2 = 0.09
    f = np.load('GenData/'+filename)

    t_start = np.argmax(np.sum(f, axis=(0,1)))
    f = f[:, :, t_start:]
    f_tot = np.sum(f, axis=(0,1))

    l1 = 0.05
    l2 = 0.09
    nx, ny, n_time = np.shape(f)

    iters = 500

    cs = np.zeros((nx, ny, iters+1))

    for jx in range(nx):
        for jy in range(ny):

            f_pix = f[jx, jy, :]

            X_pix = f2X(f_pix, 1)

            n = len(X_pix)
            e1 = exp(X_pix, l1)
            e2 = exp(X_pix, l2)

            cs[jx, jy, 0] = 0.5

            if (n == 0):
                cs[jx, jy, :] = np.full(iters+1, 0.5)

            else:
                for i in range(iters):
                    cs[jx, jy, i+1] = new_concentration_EM(cs[jx, jy, i], e1, e2, n)
    
    c = cs[:, :, iters]
    cs_avg.append(np.average(c))
    cs_std.append(np.std(c))
    cs_true.append(c_true)

plt.figure()

plt.errorbar(cs_true, cs_avg, yerr=cs_std, fmt='.')
plt.plot(cs_true, cs_true, '--')
plt.xlabel(r'$c_{gen}$', fontsize=12)
plt.ylabel(r'$c_{MLE}$', fontsize=12)
