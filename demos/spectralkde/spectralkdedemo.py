# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:03:03 2019

@author: suraj
"""

import matplotlib.pyplot as plt
from utils.spectralkde import SpectralKDE
from utils.simulation import CPPSimulation
from utils.distributions import Gaussian, MixtureOfGaussians
from utils.charfunctions import wand_kernel

def generate_plot1():
    cpp = CPPSimulation(1, Gaussian())
    jumps = cpp.observe_jumps(1, n_obs=5000)
    spectralkde = SpectralKDE(0.1, 16384, 0.1)
    density = spectralkde.compute_density(jumps, wand_kernel, 1)
    x = spectralkde.compute_x()
    
    plt.plot(x, density)
    plt.xlim(-4, 4)
    plt.ylim(0, 0.5)
    plt.plot(x, Gaussian().pdf(x))
    plt.show()

def generate_plot2():
    mog = MixtureOfGaussians([0.3, 0.7], [2, -1], [1, 1/9])
    cpp = CPPSimulation(1, mog)
    jumps = cpp.observe_jumps(1, n_obs=5000)
    spectralkde = SpectralKDE(0.1, 16384, 0.1)
    density = spectralkde.compute_density(jumps, wand_kernel, 1)
    x = spectralkde.compute_x()
    
    plt.plot(x, density)
    plt.xlim(-5, 5)
    plt.plot(x, mog.pdf(x))
    plt.show()




