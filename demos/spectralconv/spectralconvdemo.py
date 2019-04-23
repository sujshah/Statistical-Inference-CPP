# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:28:30 2019

@author: suraj
"""

import matplotlib.pyplot as plt
from utils.spectralconv import SpectralConv
from utils.simulation import CPPSimulation
from utils.distributions import Gaussian, MixtureOfDistribution, Laplace, Cauchy
import numpy as np
plt.rcParams['font.size'] = 12

def generate_plot1():
    cpp = CPPSimulation(rate=0.3, distribution=Gaussian())
    jumps = cpp.observe_jumps(0.4, n_obs=5000)
    spectralconv = SpectralConv(0.15, 4096, 10)
    density = spectralconv.compute_density(jumps, 0.4, 0.3)
    x = spectralconv.compute_x()
    
    y = np.arange(-5, 5, 0.01)
    plt.plot(x, density)
    plt.xlim(-4, 4)
    plt.ylim(0, 0.5)
    plt.plot(y, Gaussian().pdf(y))
    plt.show()
    
def generate_plot2():
    true_dist = MixtureOfDistribution(Gaussian, [0.3, 0.7], [-2, 1], [1, 1])
    cpp = CPPSimulation(rate=0.3, distribution=true_dist)
    jumps = cpp.observe_jumps(0.4, n_obs=5000)
    spectralconv = SpectralConv(0.15, 16384, 10)
    density = spectralconv.compute_density(jumps, 0.4, 0.3)
    x = spectralconv.compute_x()
    
    y = np.arange(-5, 5, 0.01)
    plt.plot(x, density)
    plt.xlim(-4, 4)
    plt.ylim(0, 0.5)
    plt.plot(y, true_dist.pdf(y))
    plt.show()

def generate_plot3():
    true_dist = Laplace(0, 1)
    cpp = CPPSimulation(rate=0.3, distribution=true_dist)
    jumps = cpp.observe_jumps(0.4, n_obs=5000)
    spectralconv = SpectralConv(0.15, 4096, 10)
    density = spectralconv.compute_density(jumps, 0.4, 0.3)
    x = spectralconv.compute_x()
    
    y = np.arange(-5, 5, 0.01)
    plt.plot(x, density)
    plt.xlim(-4, 4)
    plt.ylim(0, 0.5)
    plt.plot(y, true_dist.pdf(y))
    plt.show()

def generate_plot4():
    true_dist = Cauchy(0, 1)
    cpp = CPPSimulation(rate=0.3, distribution=true_dist)
    jumps = cpp.observe_jumps(0.4, n_obs=5000)
    spectralconv = SpectralConv(0.15, 4096, 10)
    density = spectralconv.compute_density(jumps, 0.4, 0.3)
    x = spectralconv.compute_x()
    
    y = np.arange(-5, 5, 0.01)
    plt.plot(x, density)
    plt.xlim(-4, 4)
    plt.ylim(0, 0.5)
    plt.plot(y, true_dist.pdf(y))
    plt.show()

generate_plot3()
