"""
Created on Mon Mar 25 17:21:05 2019

@author: suraj
"""
import numpy as np

class CPPSimulation:
    """
    Implements a Simulation of CPP using exponential interarrival times
    property of a Poisson process. 
    """
    
    def __init__(self, rate, distribution):
        """
        Initialises a CPP Simulation with required parameters.
        :param rate: the rate of the underlying Poisson process.
        :param distribution: the distribution of the terms in the CPP.
        """
        
        self.rate = rate
        self.distribution = distribution
    
    def simulate(self, T):
        """
        Performs the simulation of the CPP up to time T and return the array
        of (time, value) tuples of the jump points.
        :param T: time up to which we simulate the CPP.
        :return: array of tuples (time, value) of the points at which the CPP
                 jumps.
        """
        
        times = np.cumsum(np.random.exponential(
                scale=1.0/self.rate, size=int(self.rate*T*5)))
        times = times[times < T]
        n = len(times)
        values = np.cumsum(self.distribution.sample(n))
        res = list(zip(times, values))
        
        #append (T, value) to end of res
        res.append((T, values[-1]))
        #insert (0,0) at start
        res.insert(0, (0.0, 0.0))
        
        return np.array(res)
    
    def observe(self, delta, n_obs):
        """
        Collects our observations of the CPP at discrete points of equal
        distance \Delta apart.
        :param delta: the separation distance of the observed points.
        :param n_obs: the number of observations we want to obtain. 
        :return: array of discrete points at which we observe the CPP.
        """
        
        simulation = self.simulate(delta*n_obs)
        obs, current_time = [], 0
        for i in range(1, n_obs+1):
            while simulation[current_time][0] < i*delta: current_time += 1
            obs.append(simulation[current_time-1][1])
        return np.array(obs)
    
    def observe_jumps(self, delta, n_obs):
        """
        Collects our observations of the CPP and then computes the jump sizes
        for all non-zero increments. We ensure we return n_obs number of jumps.
        :param delta: the separation distance of the observed points.
        :param n_obs: the number of non-zero jumps we want to obtain.
        :return: array of jumps of size n_obs
        """
        
        observations = self.observe(delta, n_obs*5)
        jumps = (observations[1:] - observations[:-1])
        non_zero_jumps = jumps[jumps != 0]
        
        if n_obs < non_zero_jumps.size:
            return non_zero_jumps[:n_obs] 
        return non_zero_jumps
        
        
        
        
        


    
    