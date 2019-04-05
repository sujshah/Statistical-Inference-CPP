# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:39:10 2019

@author: suraj
"""

import numpy as np
from scipy.special import factorial
import simulation, distributions

jumps = simulation.CPPSimulation(1, distributions.Gaussian()).observe_jumps(1)
