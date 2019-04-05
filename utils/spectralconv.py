"""
Created on Wed Mar 27 13:29:16 2019

@author: sujshah
"""
import numpy as np
from charfunctions import char_emp

class SpectralConv:
    """
    Implements the Spectral approach via computing kernel density estimates
    of the convolutions of the density.
    """
    
    def __init__(self, bandwidth, fourier_N, n_conv):
        """
        Initialises the tuning parameters for the kernel density estimation
        via convolutions.
        :param bandwidth: bandwidth parameter for the kernel.
        :param fourier_N: number of fourier terms. Preferably some power of 2.
        :param n_conv: the number of convolutions to take into consideration
                       after truncating the series.
        """
        
        self.h = bandwidth
        self.N = fourier_N
        self.eta = np.pi/(self.h*(self.N - 1))
        self.delta = 2*self.h*(self.N - 1)/self.N
        self.n_conv = n_conv
    
    def compute_convolution(self, m, jumps, rate):
        """
        Computes the estimator for the m-convolution of the density along an 
        array of points x given the rate of the underlying Poisson process and
        the jump observations.
        :param m: the number of times the density has convolved with itself.
        :param jumps: the non-zero jump observations.
        :param rate: the rate of the underlying Poisson process.
        :return: estimate for the m-convolution of the density.
        """
        
        steps = np.arange(self.N)
        
        psi1 = ((char_emp(jumps, steps*self.eta)**m)*
                np.exp(1j*steps*np.pi))*self.eta
        psi2 = ((char_emp(jumps, -steps*self.eta)**m)*
                np.exp(-1j*steps*np.pi))*self.eta)
        
        f1 = np.fft.fft(psi1)/(2*np.pi)
        f2 = self.N*np.fft.ifft(psi2)/(2*np.pi)
        
        return f1 + f2
    
    def compute_density(self, jumps, separation, rate):
        """
        Computes the estimator for the density via suitable summatation up to
        n_conv of the convolutions.
        :param jumps: the non-zero jump observations.
        :param separation: the separation time between the observations.
        :param rate: the rate of the underlying Poisson process.
        :return: density estimate evaluated at points 
                 $(-frac{N\delta}{2} + \delta*i)$ where i=0,1,...,N.
        """
        
        index = np.arange(1, self.n_conv + 1)
        convs = np.asarray([self.compute_convolution(m, jumps, rate) for
                            m in index])
        coeffs = (((-1)**(index + 1))*((np.exp(rate*separation) - 1)**index)/
                  (index*rate*separation))
        
        return np.sum(coeffs*convs, axis=0)        
        
    def compute_x(self):
        """
        Computes the array of points x that we have computed the density over.
        :return: array of points x of length fourier_N.
        """
        
        return np.asarray([-self.N*self.delta/2 + self.delta*j for 
                       j in range(self.N)])
