# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:46:48 2019

@author: suraj
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm


#We simulate a CPP up to time T using the fact that the arrival times
#for a Poisson process are independent exponentially
#distributed random variables with rate \lambda
def cpp_sim(T, rate, density):
    currentTime = 0.0
    times = np.array([currentTime])
    
    while currentTime < T:
        currentTime += np.random.exponential(scale = 1.0/rate)
        times = np.append(times, currentTime)
    
    n = len(times) - 1
    values = np.insert(
    np.cumsum(density(size = n)), 0, 0.0
    )
    return np.append(times[:-1], T), np.append(values[:-1], values[:-1][-1])

times, values = cpp_sim(100, 1.0, np.random.normal)
plt.step(times, values)    

def non_zero_jumps(n, delta, rate, density):
    T = n*delta
    times, values = cpp_sim(T, rate, density)
    plt.step(times, values)
    obs = np.array([])
    currentTime = 0
    for i in range(1, n + 1):
        while times[currentTime] < i*delta:
            currentTime += 1
        obs = np.append(obs, values[currentTime - 1])
        
    return np.array([obs[i] - obs[i-1] for i in range(1, n) 
    if obs[i] != obs[i-1]])
    
    



#True density f0 of the jumps
def f0(size):
    return 0.8*np.random.normal(loc=2.0, scale=1/1.0, size=size)
    + 0.2*np.random.normal(loc=-1.0, scale=1/1.0, size=size)

def func():
    #We build a MCMC algorithm using a data augmentation scheme
    #as explained in Gugushvili
    
    #sample size
    n = 500
    
    #segment length
    delta = 1.0
    
    #rate of CPP
    lambda0 = 1.0
    #rates of each type
    J = 2
    psi0 = [0.8, 0.2]
    
    #Prior Hyperparameters
    alpha, beta = (1.0, 1.0), (1.0, 1.0)
    xi = np.array([0.0, 0.0])
    k = 1.0
    
    #Priors
    psi = np.random.gamma(alpha[0], 1/beta[0], J)
    tau = np.random.gamma(alpha[1], 1/beta[1])
    mu = np.random.normal(xi, 1/(tau*k), J)
    
    #Hierarchical model and data augmentation scheme
    
    #We simulate a CPP with density f0 and then record the
    #segments of non-zero jump sizes
    T = n*delta
    times, values = cpp_sim(T, lambda0, f0)
    plt.step(times, values)
    obs = np.array([])
    currentTime = 0
    for i in range(1, n + 1):
        while times[currentTime] < i*delta:
            currentTime += 1
        obs = np.append(obs, values[currentTime - 1])
        
    nonZeroObs = np.array([obs[i] - obs[i-1] for i in range(1, n) 
    if obs[i] != obs[i-1]])
    
    
    #Updating segments
    numberOfSegments = len(nonZeroObs)
    
    #initial a
    a = np.repeat(np.array([[0.]*J]), repeats=numberOfSegments, axis=0)
    JumpSegments = np.repeat(1, numberOfSegments)
    count = 0
    
    for _ in range(5000):
        for i in range(numberOfSegments):
            proposalSegmentJumpSize = np.random.poisson(lam=sum(psi)*delta)
            while proposalSegmentJumpSize == 0: 
                proposalSegmentJumpSize = np.random.poisson(lam=sum(psi)*delta)
            
            proposalJumpSizeTypes = np.random.multinomial(
                    proposalSegmentJumpSize, psi)
        
            acceptanceProb = norm.pdf(nonZeroObs[i], proposalJumpSizeTypes.dot(mu), 
                                      proposalSegmentJumpSize/tau)/norm.pdf(
                                              nonZeroObs[i], a[i].dot(mu),
                                              JumpSegments[i]/tau)
            
            randomNumber = np.random.uniform()
            if randomNumber <= acceptanceProb:
                a[i] = proposalJumpSizeTypes
                JumpSegments[i] = proposalSegmentJumpSize
        
        
        
        #Update Parameters
        s = np.sum(a, axis=0)
        psi = np.random.gamma(alpha[0] + s, 1/(beta[0] + T))
        
        Pscribble = np.zeros(shape = (J, J))
        for j in range(J):
            for k in range(J):
                Pscribble[j][k] = np.sum((a[:,j]*a[:,k])/JumpSegments)
        
        P = k*np.identity(J) + Pscribble
        invP = np.linalg.inv(P)
        q = np.zeros(J)
        for j in range(J):
            q[j] = k*xi[j] + np.sum((a[:,j]*nonZeroObs)/JumpSegments)
        
        R = k*np.sum(xi**2) + np.sum((nonZeroObs**2)/JumpSegments)
        
        tau = np.random.gamma(alpha[1] + nonZeroObs.size/2, 1/(beta[1] + R - 
                              np.matmul(q, invP).dot(q)/2))
        
        mu = np.random.multivariate_normal(np.matmul(invP, q), (1/tau)*invP)
    
    print(mu)
    print(tau)
    print(psi)

