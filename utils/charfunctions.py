"""
Created on Wed Mar 27 13:05:04 2019

@author: sujshah
"""
import numpy as np

"""
Implements all the characteristic functions we need for our computations.
"""

def char_emp(jumps, x):
    """
    Evaluates the empirical characteristic function given the jump
    observations along an array of points x.
    :param jumps: the non-zero jump observations.
    :param x: array of points at which to compute the characteristic fn.
    :return: empirical characteristic function evaluated at points x.
    """
    
    return np.asarray([np.sum(np.exp(1j*xi*jumps)) for xi in x])/jumps.size

def wand_kernel(x):
    """
    Evaluates the characteristic function from Wand along an array of 
    points x.
    :param x: array of points at which to compute the characteristic fn
    :return: characteristic function evaluated at points x.
    """
    
    return np.asarray([(1-xi**2)**3 if np.abs(xi) < 1 else 0 for xi in x])
