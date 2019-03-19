# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:36:26 2019

@author: suraj
"""
from cpp_sim import cpp_sim, non_zero_jumps, mixture_normals
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

plt.rcParams.update({'font.size': 22})

times, values = cpp_sim(100, 0.3, np.random.normal)
plt.step(times, values)

def kernel(t):
    return (48*t*(t**2 - 1)*np.cos(t) - 144*(2*t**2 - 5)*np.sin(t))/(np.pi*t**7)

def characteristic_kernel(t):
    return np.array([(1-ti**2)**3 if np.abs(ti) < 1 else 0 for ti in t])

n = 10000
bandwidth = 0.14
lamb = 0.3

Z = non_zero_jumps(n, 1, 0.3, np.random.normal)
Z = Z[:1000]

eta = 0.01
N = 16384
delta = 2*np.pi/(N*eta)
def v(eta, N):
    return np.array([eta*j for j in range(N)])

v = v(eta, N)    

def characteristic_emp(Z, t):
    n = Z.size
    return np.array([np.sum(np.exp(1j*ti*Z))/n for ti in t])

def characteristic_gnh(Z, t):
    return characteristic_emp(Z, t)*characteristic_kernel(bandwidth*t)

def psi(v, lamb, Z):
    return np.log((np.exp(lamb) - 1)*characteristic_gnh(Z, v) + 1)

def compute_f1(v, lamb, Z):
    fft = np.fft.fft(psi(v, lamb, Z)*np.exp(1j*v*N*delta/2)*eta)
    return fft/(2*np.pi*lamb)

f1 = compute_f1(v, lamb, Z)
x = -N*delta/2 + delta*(np.arange(N))

def psi2(v, lamb, Z):
    return np.log((np.exp(lamb) - 1)*characteristic_emp(Z, -v)*characteristic_kernel(bandwidth*v) + 1)

def compute_f2(v, lamb, Z):
    ifft = np.fft.ifft(psi2(v, lamb, Z)*np.exp(-1j*v*N*delta/2)*eta)
    return N*ifft/(2*np.pi*lamb)
    
f2 = compute_f2(v, lamb, Z)
plt.scatter(x, f1 + f2)
plt.xlim(-4, 4)
plt.ylim(0, 0.42)
plt.plot(x, mlab.normpdf(x, 0, 1))
plt.show()


n = 10000
bandwidth = 0.1
lamb = 0.3

Z = non_zero_jumps(n, 1, 0.3, mixture_normals)
Z = Z[:1000]

np.mean(mixture_normals(1000))

eta = 0.01
N = 16384
delta = 2*np.pi/(N*eta)

f1 = compute_f1(v, lamb, Z)
f2 = compute_f2(v, lamb, Z)
x = -N*delta/2 + delta*(np.arange(N))

plt.scatter(x, f1 + f2)
plt.xlim(-4, 4)
plt.ylim(0, 0.5)
plt.plot(x, (2/3)*mlab.normpdf(x, 0, 1) + (1/3)*mlab.normpdf(x, 3/2, 1/3))
plt.show()

plt.scatter(x, f1)
plt.xlim(-4,4)
plt.ylim(0, 0.5)
plt.show()

def trapezoid(v):
    pass