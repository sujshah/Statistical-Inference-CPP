"""
Created on Fri Mar 29 15:58:05 2019

@author: suraj
"""
import numpy as np
from distributions import Gaussian, ZeroTruncatedPoission
from time import sleep
np.set_printoptions(precision=3)

class BayesianMoG:
    """
    Implements the Bayesian approach of density estimation of a mixture of
    Gaussians of fixed size using a data augmentation scheme.
    """
    
    def __init__(self, n_components, n_iterations, jumps, T, delta,
                 burn_in=5000, store_freq=1, hyperparameters=None):
        """
        Initialises the tuning parameters of the MCMC algorithm and the 
        non-zero jump observations needed for drawing from the posterior.
        :param n_components: number of components in our mixture.
        :param n_iterations: number of iterations of our MCMC.
        :param jumps: the non-zero jump observations of the CPP.
        :param T: the time up to which we observe the CPP.
        :param delta: the separation time between the observations.
        :param burn_in: MCMC burn in.
        :param store_freq: the frequency at which we store iterations.
        :param hyperparameters: hyperparameters for our parameters we want to
                                 obtain inference on.
        """
        
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.store_freq = store_freq
        self.acceptance_rate = 0.
        
        self.mixing_coeffs = np.zeros((self.n_iterations//self.store_freq,
                                       self.n_components))
        self.means = np.zeros((self.n_iterations//self.store_freq,
                               self.n_components))
        self.precision = np.zeros(self.n_iterations//self.store_freq)
        
        self.jumps = jumps
        self.n_segments = self.jumps.size
        self.auxiliary = np.ones((self.n_segments, self.n_components))
        self.T = T
        self.delta = delta
        
        if hyperparameters is None:
            hyperparameters = {'mix_shape' : np.ones(self.n_components),
                               'mix_rate' : 1,
                               'precision_shape' : 1,
                               'precision_rate' : 1,
                               'means_loc' : np.zeros(self.n_components),
                               'means_scale' : 1}
        
        self.hyparam = hyperparameters
        self.currentmixing_coeffs = np.zeros(self.n_components)
        self.currentmeans = np.zeros(self.n_components)
        self.currentprecision = np.zeros(1)
    
    def performMCMC(self):
        """
        Performs the MCMC algorithm to sample from the posterior.
        """
        
        self.initialise_parameters()
        accept_count = 0
        
        print(f'Performing MCMC with {self.n_iterations} steps.')
        print('Initial parameter values are:')
        print(f'Mixing Coeffs: {self.mixing_coeffs[0]}')
        print(f'Means: {self.means[0, :]}')
        print(f'Precision: {self.precision[0]:.3f}')
        sleep(4)
        
        for it in range(1, self.n_iterations):
            accept_count += self.update_segments(it)
            self.update_parameters(it)
            self.acceptance_rate = accept_count/it
            
            if it % self.store_freq == 0:
                store_it = it//self.store_freq
                self.mixing_coeffs[store_it] = self.currentmixing_coeffs
                self.means[store_it] = self.currentmeans
                self.precision[store_it] = self.currentprecision
            
            print(f'Iteration {it}:')
            print(f'Mixing Coeffs: {self.currentmixing_coeffs}')
            print(f'Means: {self.currentmeans}')
            print(f'Precision: {self.currentprecision:.3f}')
            print(f'Acceptance Rate: {self.acceptance_rate:.3f}%')
        
        print('MCMC has completed.')
        
    def initialise_parameters(self):
        """
        Initialises the parameter values using the hyperparameters.
        """
        
        self.mixing_coeffs[0] = np.random.gamma(
                self.hyparam['mix_shape'], 
                1.0/self.hyparam['mix_rate'])
        
        self.precision[0] = np.random.gamma(
                self.hyparam['precision_shape'], 
                1.0/self.hyparam['precision_rate'])
        
        self.means[0] = np.random.normal(
                self.hyparam['means_loc'],
                np.sqrt(1.0/(self.precision[0]*self.hyparam['means_scale'])))
        
        self.currentmixing_coeffs = self.mixing_coeffs[0]
        self.currentmeans = self.means[0]
        self.currentprecision = self.precision[0]
    
    def computeQPR(self, segment_sums):
        """
        Computes the P, Q, R matrix/vector/scalar for use in updating the 
        parameters. See Essay for more information.
        :param segment_sums: the jump sums for each segment in the auxiliary
                             variable.
        :return: vector Q, matrix P, scalar R.
        """
        
        Q = ((self.hyparam['means_scale']*
             self.hyparam['means_loc']) +
             np.matmul(self.auxiliary.transpose(), self.jumps/segment_sums))
        
        P = ((self.hyparam['means_scale']*
             np.identity(self.n_components)) +
             np.matmul(self.auxiliary.transpose(),
                       self.auxiliary/segment_sums[:, np.newaxis]))
        
        R = ((self.hyparam['means_scale']*
              np.sum(self.hyparam['means_loc']**2)) + 
             np.sum((self.jumps**2)/segment_sums))
        
        return Q, P, R
    
    def update_parameters(self, it):
        """
        Samples the parameter values condiitonal on the auxiliary variable
        using the conjugate prior formulas.
        :param it: the MCMC iteration number.
        """
        
        component_sums = np.sum(self.auxiliary, axis=0)
        segment_sums = np.sum(self.auxiliary, axis=1)  
        Q, P, R = self.computeQPR(segment_sums)
        invP = np.linalg.inv(P)
        
        self.currentmixing_coeffs = np.random.gamma(
                self.hyparam['mix_shape'] + component_sums, 
                1.0/(self.hyparam['mix_rate'] + self.T))
        
        self.currentprecision = np.random.gamma(
                self.hyparam['precision_shape'] + self.n_segments/2,
                1.0/(self.hyparam['precision_rate'] + 
                     (R - np.dot(Q, np.matmul(invP, Q)))/2.0))
        
        self.currentmeans = np.random.multivariate_normal(np.matmul(invP, Q),
                  invP/self.currentprecision)

    def update_segments(self, it):
        """
        Samples the auxiliary variable using a Metropolis-Hastings step
        conditional on the updated parameter values.
        :param it: theh MCMC iteration number.
        :return: the average number of proposal acceptances over each segment.
        """
        
        accept_count = 0
        zerotrun_pois = (ZeroTruncatedPoission(
                rate=np.sum(self.currentmixing_coeffs)*self.delta)
            .sample(self.n_segments))
        
        for seg in range(self.n_segments):
            prop = zerotrun_pois[seg]
            prop_components = np.random.multinomial(
                prop, 
                self.currentmixing_coeffs/np.sum(self.currentmixing_coeffs))
            
            accept_top = Gaussian(
                    np.dot(prop_components, self.currentmeans), 
                    prop/self.currentprecision).pdf(self.jumps[seg])
            
            accept_bot = Gaussian(
                    np.dot(self.auxiliary[seg], self.currentmeans),
                    np.sum(self.auxiliary[seg])/self.currentprecision).pdf(
                            self.jumps[seg])
            
            if np.random.rand() < accept_top/accept_bot:
                accept_count += 1
                self.auxiliary[seg] = prop_components
        
        return accept_count/self.n_segments
    
    def calculate_posterior(self):
        """
        Calculates the posterior mean estimates after removing burn_in.
        :return: dictionary of posterior mean estimates for the parameters
                 required.
        """
        
        posterior = {}
        posterior['mixing_coeffs'] = np.mean(self.mixing_coeffs[self.burn_in:],
                                          axis=0)
        posterior['means'] = np.mean(self.means[self.burn_in:], axis=0)
        posterior['precision'] = np.mean(self.precision[self.burn_in:])
        
        return posterior 
        
        
        
        
        
