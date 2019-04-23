# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:03:03 2019

@author: suraj
"""

import matplotlib.pyplot as plt
from utils.spectralkde import SpectralKDE
from utils.simulation import CPPSimulation
from utils.distributions import Gaussian, MixtureOfDistribution, Laplace, Cauchy
from utils.charfunctions import wand_kernel
plt.rcParams['font.size'] = 12

def generate_plot1():
    cpp = CPPSimulation(0.3, Gaussian())
    jumps = cpp.observe_jumps(0.4, n_obs=5000)
    spectralkde = SpectralKDE(0.14, 16384, 0.01)
    density = spectralkde.compute_density(jumps, wand_kernel, 0.4)
    x = spectralkde.compute_x()
    
    plt.plot(x, density)
    plt.xlim(-4, 4)
    plt.ylim(0, 0.5)
    plt.plot(x, Gaussian().pdf(x))
    plt.legend(['Estimate', 'True'])
    plt.show()

def generate_plot2():
    mog = MixtureOfDistribution(Gaussian, [0.3, 0.7], [-2, 1], [1, 1])
    cpp = CPPSimulation(0.3, mog)
    jumps = cpp.observe_jumps(0.4, n_obs=5000)
    spectralkde = SpectralKDE(0.14, 16384, 0.01)
    density = spectralkde.compute_density(jumps, wand_kernel, 0.4)
    x = spectralkde.compute_x()
    
    plt.plot(x, density)
    plt.xlim(-5, 5)
    plt.ylim(0, 0.5)
    plt.plot(x, mog.pdf(x))
    plt.legend(['Estimate', 'True'])
    plt.show()

def generate_plot3():
    laplace = Laplace(0, 1)
    cpp = CPPSimulation(0.3, laplace)
    jumps = cpp.observe_jumps(0.4, n_obs=5000)
    spectralkde = SpectralKDE(0.1, 16384, 0.01)
    density = spectralkde.compute_density(jumps, wand_kernel, 0.4)
    x = spectralkde.compute_x()
    
    plt.plot(x, density)
    plt.xlim(-5, 5)
    plt.ylim(0, 0.6)
    plt.plot(x, laplace.pdf(x))
    plt.legend(['Estimate', 'True'])
    plt.show()

def generate_plot4():
    cauchy = Cauchy(0, 1)
    cpp = CPPSimulation(0.3, cauchy)
    jumps = cpp.observe_jumps(0.4, n_obs=5000)
    spectralkde = SpectralKDE(0.14, 16384, 0.01)
    density = spectralkde.compute_density(jumps, wand_kernel, 0.4)
    x = spectralkde.compute_x()
    
    plt.plot(x, density)
    plt.xlim(-6, 6)
    plt.ylim(0, 0.5)
    plt.plot(x, cauchy.pdf(x))
    plt.legend(['Estimate', 'True'])
    plt.show()
    

generate_plot3()

