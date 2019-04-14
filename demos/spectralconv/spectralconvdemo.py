# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:28:30 2019

@author: suraj
"""

import matplotlib.pyplot as plt
from utils.spectralconv import SpectralConv
from utils.simulation import CPPSimulation
from utils.distributions import Gaussian, MixtureOfDistribution

"""
cpp = CPPSimulation(rate=0.3, distribution=Gaussian())
jumps = cpp.observe_jumps(0.4, n_obs=5000)
spectralconv = SpectralConv(0.15, 16384, 10)
density = spectralconv.compute_density(jumps, 0.4, 0.3)
x = spectralconv.compute_x()

plt.plot(x, density)
plt.xlim(-4, 4)
plt.ylim(0, 0.5)
plt.plot(x, Gaussian().pdf(x))
plt.show()
"""

true_dist = MixtureOfDistribution(Gaussian, [0.3, 0.7], [-2, 1], [1, 1])
cpp = CPPSimulation(rate=0.3, distribution=true_dist)
jumps = cpp.observe_jumps(0.4, n_obs=5000)
spectralconv = SpectralConv(0.15, 16384, 10)
density = spectralconv.compute_density(jumps, 0.4, 0.3)
x = spectralconv.compute_x()

plt.plot(x, density)
plt.xlim(-4, 4)
plt.ylim(0, 0.5)
plt.plot(x, true_dist.pdf(x))
plt.show()

