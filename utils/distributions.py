"""
Created on Mon Mar 25 14:46:24 2019

@author: sujshah
"""
import numpy as np
import scipy.stats as stats

class Gaussian:
    """
    Implements a Gaussian distribution.
    Focus is on sampling and probability density function.
    """
    
    def __init__(self, mean=0.0, variance=1.0):
        """
        Initialises a Gaussian distribution with mean and variance parameters.
        :param mean: mean.
        :param variance: variance.
        """
        
        if mean is not None:
            self.mean = mean
            
            if variance is not None:
                self.variance = variance
            
            else:
                raise ValueError("Covariance information missing")
        else:
            raise ValueError("Mean information missing")
        
    
    def sample(self, n_samples=1):
        """
        Returns independent samples from the Gaussian distribution.
        :param n_samples: number of samples required.
        :return: samples from the Gaussian distribution.
        """
        return np.random.normal(self.mean, np.sqrt(self.variance), n_samples)
    
    def pdf(self, x):
        """
        Evaluates the Gaussian pdf at discrete values x.
        :param x: the input array to evaluate the Gaussian pdf at.
        :return: Gaussian pdf.
        """
        return stats.norm.pdf(x, self.mean, np.sqrt(self.variance))   
        

class MixtureOfGaussians:
    """
    Implements a mixture of Gaussians.
    Focus is on sampling and probability density function.
    """
    
    def __init__(self, mixing_coeffs, means, variances):
        """
        Creates a MoG using a list of means and variances with corresponding
        mixing coefficients.
        :param mixing_coeffs: mixing coefficients
        :param means: list of means
        :param variances: list of variances
        """
        
        if means is not None:
            
            if variances is not None:
                self.gaussians = [Gaussian(mean=mean, variance=variance) for 
                                  mean, variance in zip(means, variances)]
            else:
                raise ValueError("Variance information missing.")
        else:
            raise ValueError("Mean information missing.")
        
        self.mixing_coeffs = mixing_coeffs
        self.n_components = len(self.gaussians)
    
    def sample(self, n_samples=1):
        """
        Returns independent samples from the MoG distribution.
        :param n_samples: number of samples required.
        :return: samples from the MoG distribution.
        """
    
        ii = np.random.choice(a=np.arange(self.n_components), size=n_samples,
                              p=self.mixing_coeffs)
        ns = [np.sum((ii == i).astype(int)) for i in range(self.n_components)]
        samples = [gaussian.sample(n) for gaussian, n in 
                   zip(self.gaussians, ns)]
        samples = np.concatenate(samples)
        np.random.shuffle(samples)
        
        return samples
    
    def pdf(self, x):
        """
        Evaluates MoG pdf at discrete values x.
        :param x: the input array to evaulate the MoG pdf at.
        :return: MoG pdf.
        """
        pdfs = np.array([gaussian.pdf(x) for gaussian in self.gaussians])
        
        return np.dot(self.mixing_coeffs, pdfs)
    
    
        
        
            
        
        
        
        
        
        
    
    