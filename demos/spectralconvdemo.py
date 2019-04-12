# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:28:30 2019

@author: suraj
"""

import matplotlib.pyplot as plt
from spectralconv import SpectralConv
from simulation import CPPSimulation
from distributions import Gaussian

cpp = CPPSimulation(rate=0.3, distribution=Gaussian())
jumps = cpp.observe_jumps(0.4, n_obs=5000)
spectralconv = SpectralConv(0.1, 16384, 10)
density = spectralconv.compute_density(jumps, 0.4, 0.3)
x = spectralconv.compute_x()

plt.plot(x, density)
plt.xlim(-10, 10)
plt.ylim(-1, 1)
plt.plot(x, Gaussian().pdf(x))
plt.show()
