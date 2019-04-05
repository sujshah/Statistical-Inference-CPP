"""
Created on Mon Mar 25 17:21:05 2019

@author: sujshah
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
        
    def simulate_pois(self, T):
        """
        Performs the simulation of a Poisson process of intensity rate up to
        time T and returns the 1Darray of times of the jump
        points.
        :param T: time up to which we simulate the Poisson process.
        :return: 1Darray of times at which the 
                 Poisson process jumps.
        """
        
        times = np.cumsum(np.random.exponential(
                scale=1.0/self.rate, size=int(self.rate*T*5)))
        
        return np.insert(times[times < T], 0, 0.0)
        
    
    def simulate_cpp(self, T):
        """
        Performs the simulation of the CPP up to time T and returns the 2Darray
        of (time, value) pairs of the jump points.
        :param T: time up to which we simulate the CPP.
        :return: 2Darray of pairs (time, value) of the points at which the CPP
                 jumps.
        """
        
        times = self.simulate_pois(T)
        n = len(times) - 1
        values = np.insert(np.cumsum(self.distribution.sample(n)), 0, 0.0)
        res = list(zip(times, values))
        
        #append (T, value) to end of res
        res.append((T, values[-1]))
        
        return np.array(res)
    
    def observe_cpp(self, delta, T):
        """
        Collects our observations of the CPP at discrete points of equal
        distance \Delta apart.
        :param delta: the separation distance of the observed points.
        :param T: the time up to which we collect observations. 
        :return: array of discrete points at which we observe the CPP.
        """
        
        simulation = self.simulate_cpp(T)
        obs, current_time, i = [0.], 0, 1
        
        while i*delta < T:
            while simulation[current_time][0] < i*delta: current_time += 1
            obs.append(simulation[current_time-1][1])
            i += 1
        
        return np.array(obs)
    
    def observe_jumps(self, delta, n_obs=None, T=None):
        """
        Collects our observations of the CPP and then computes the jump sizes
        for all non-zero increments. If n_obs is specified, we ensure we return
        n_obs number of jumps. If T is specified, we return the non-zero
        increments up to time T. If both are specified, we take the n_obs case.
        :param delta: the separation distance of the observed points.
        :param n_obs: the number of non-zero jumps we want to obtain.
        :param T: the time T until which we collect non-zero jumps.
        :return: array of non-zero jumps.
        """
        
        if all(v is None for v in {n_obs, T}):
            raise ValueError("Expected either n_obs or T.")
        
        if n_obs is not None:
            T = delta*n_obs*5
        
        observations = self.observe_cpp(delta, T)
        jumps = (observations[1:] - observations[:-1])
        non_zero_jumps = jumps[jumps != 0]
        
        if n_obs is not None:
            return non_zero_jumps[:min(n_obs, non_zero_jumps.size)] 
        return non_zero_jumps
        
        
        
        
        


    
    