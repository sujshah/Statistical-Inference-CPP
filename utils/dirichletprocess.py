# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:53:24 2019

@author: suraj
"""
import numpy as np
from matplotlib import pyplot as plt
from distributions import MixtureOfGaussians, Gaussian

def stick_breaking_process(num_weights, alpha):
    betas = np.random.beta(1,alpha, size=num_weights)
    log_betas = np.log(betas)
    log_betas[1:] = log_betas[1:] + np.cumsum(np.log(1 - betas[:-1]))
    return np.exp(log_betas)

dp1 = stick_breaking_process(10, 0.001)
print(np.sum(dp1))
plt.bar(np.arange(1, 11), dp1)

dp2 = stick_breaking_process(10, 2)
#plt.bar(np.arange(1, 11), dp2)

dp3 = stick_breaking_process(10, 5)
#plt.bar(np.arange(1, 11), dp3)

def generate_gaussian_mixture(num_weights, alpha):
    means = Gaussian(0, 5).sample(num_weights)
    variances = np.random.gamma(1, size=num_weights)
    mixing_coeffs = stick_breaking_process(num_weights, alpha)
    return MixtureOfGaussians(mixing_coeffs, means, variances)

"""
mog = generate_gaussian_mixture(10, 1)
x = np.arange(-10, 10, 0.1)
plt.plot(x, mog.pdf(x))
"""
    