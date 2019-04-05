# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:30:31 2019

@author: suraj
"""

from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp

from theano import tensor as tt

import simulation, distributions

jumps = simulation.CPPSimulation(0.3, 
                                 distributions.MixtureOfGaussians([0.8, 0.2], [0, 3/2], [1, 1/9])).observe_jumps(1, 5000)
delta = 1

def compute_pois(psi):
    return tt.shared(np.tile(psi, (5000, 1)))

def compute_sums(auxiliary):
    return tt.as_tensor_variable(np.sum(auxiliary, axis=1))

with pm.Model() as model:
    psi = pm.Gamma('psi', 1., 1., shape=2)
    tau = pm.Gamma('tau', 1., 1.)
    mu = pm.MvNormal('mu', np.zeros(5000), np.identity(5000)/tau, shape=5000)
    
    #pois = pm.Deterministic('pois', compute_pois(psi))
    #auxiliary = pm.Poisson('auxiliary', delta, shape=(5000, 2))
    
    #sums = pm.Deterministic('sums', compute_sums(auxiliary))
    
    obs = pm.Normal('obs', mu, tau=tau, observed=jumps)

with model:
    trace = pm.sample(100)
    
with pm.Model() as model:
    growth = pm.Normal('growth', mu=0.12, sd=0.03, shape=(10,1000))

with model:
    trace = pm.sample(100)
