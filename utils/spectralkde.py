"""
Created on Mon Mar 25 19:42:29 2019

@author: suraj
"""
import numpy as np

class SpectralKDE:
    """
    Implements the Spectral approach via kernel density estimation.
    """
    
    def __init__(self, bandwidth, fourier_N, fourier_stepsize):
        """
        Initialises the tuning parameters for the kernel density estimation.
        :param bandwidth: bandwidth parameter for the kernel.
        :param fourier_N: number of fourier terms. Preferably some power of 2.
        :param fourier_stepsize: step size of the discrete fourier sum.
        """
        
        self.h = bandwidth
        self.N = fourier_N
        self.eta = fourier_stepsize
        self.delta = 2*np.pi/(self.N*self.eta)
    
    def char_emp(self, jumps, x):
        """
        Evaluates the empirical characteristic function given the jump
        observations along an array of points x.
        :param jumps: the non-zero jump observations.
        :param x: array of points at which to compute the characteristic fn.
        :return: empirical characteristic function evaluated at points x.
        """
        
        return np.asarray([np.sum(np.exp(1j*xi*jumps)) for xi in x])/jumps.size
    
    @staticmethod
    def char_kernel(x):
        """
        Evaluates the characteristic function from Wand along an array of 
        points x.
        :param x: array of points at which to compute the characteristic fn
        :return: characteristic function evaluated at points x.
        """
        
        return np.asarray([(1-xi**2)**3 if np.abs(xi) < 1 else 0 for xi in x])
    
    def compute_density(self, jumps, char_kernel, rate):
        """
        Computes the density estimate f along an array of points x given
        the characteristic function of the kernel, the rate of the 
        underlying Poisson process and the jump observations.
        :param jumps: the non-zero jump observations.
        :param char_kernel: characteristic function of the kernel.
        :param rate: rate of the underlying Poisson process.
        :return: density estimate evaluated at points 
                 $(-frac{N\delta}{2} + \delta*i)$ where i=0,1,...,N.
        """
        
        steps = np.asarray([i*self.eta for i in range(self.N)])
        
        psi1 = np.log((np.exp(rate) - 1)*
                      char_kernel(self.h*steps)*
                      self.char_emp(jumps, steps) + 1)
        psi2 = np.log((np.exp(rate) - 1)*
                      char_kernel(self.h*steps)*
                      (self.char_emp(jumps, -steps)) + 1)
        
        f1 = np.fft.fft(psi1*np.exp(
                1j*steps*self.N*self.delta/2)*self.eta)/(2*np.pi*rate)
        f2 = self.N*np.fft.ifft(psi2*np.exp(
                -1j*steps*self.N*self.delta/2)*self.eta)/(2*np.pi*rate)
        
        return f1 + f2
    
    def compute_x(self):
        """
        Computes the array of points x that we have computed the density over.
        :return: array of points x of length fourier_N.
        """
        
        return np.asarray([-self.N*self.delta/2 + self.delta*j for 
                           j in range(self.N)])


        
        
        