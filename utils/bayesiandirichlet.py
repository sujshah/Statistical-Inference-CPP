"""
Created on Fri Mar 29 15:58:05 2019

@author: suraj
"""
import numpy as np
from distributions import Gaussian, ZeroTruncatedPoission
from time import sleep
np.set_printoptions(precision=3)

class BayesianDP:
    """
    Implements the Bayesian approach of density estimation of a mixture of
    Gaussians of fixed size using a data augmentation scheme.
    """
    
    def __init__(self, n_components, n_iterations, jumps, T, delta, rate,
                 burn_in=5000, store_freq=1, hyperparameters=None):
        """
        Initialises the tuning parameters of the MCMC algorithm and the 
        non-zero jump observations needed for drawing from the posterior.
        :param n_components: number of components in our mixture.
        :param n_iterations: number of iterations of our MCMC
        :param jumps: the non-zero jump observations of the CPP.
        :param T: the time up to which we observe the CPP.
        :param delta: the separation time between the observations.
        :param rate: the intensity of the CPP.
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
        
        self.alpha = 0
        self.betas = np.zeros(self.n_components)
        self.mixing_coeffs = np.zeros((self.n_iterations//self.store_freq,
                                       self.n_components))
        
        self.means = np.zeros((self.n_iterations//self.store_freq,
                               self.n_components))
        self.precision = np.zeros(self.n_iterations)
        
        self.metropolis = None
        
        self.jumps = jumps
        self.n_segments = self.jumps.size
        self.auxiliary = np.ones((self.n_segments, self.n_components))
        self.T = T
        self.delta = delta
        self.rate = rate
        
        if hyperparameters is None:
            hyperparameters = {'beta_shape' : 1,
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
        self.initialise_mixing_coeffs()
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
            #self.propose_move()
            self.update_mixing_coeffs(it)
            self.update_parameters(it)
            self.acceptance_rate = accept_count/it
            
            if it % self.store_freq == 0:
                self.mixing_coeffs[it] = self.currentmixing_coeffs
                self.means[it] = self.currentmeans
                self.precision[it] = self.currentprecision
            
            print(f'Iteration {it}:')
            print(f'Mixing Coeffs: {self.currentmixing_coeffs}')
            print(f'Means: {self.currentmeans}')
            print(f'Precision: {self.currentprecision:.3f}')
            print(f'Acceptance Rate: {self.acceptance_rate:.3f}%')
        
        print('MCMC has completed.')
    
    def compute_stick_breaking_weights(self, betas):
        log_betas = np.log(betas)
        log_betas[1:] = log_betas[1:] + np.cumsum(np.log(1 - betas[:-1]))
        return np.exp(log_betas)

    def initialise_mixing_coeffs(self):
        """
        Initialises the betas and computes the mixing coefficients using 
        stick breaking.
        """
        self.alpha = np.random.gamma(self.n_components + 1, 1)
        
        self.betas = np.random.beta(1, self.alpha, 
                                     size=self.n_components)
        self.mixing_coeffs[0] = self.compute_stick_breaking_weights(
                self.betas)
        
        self.currentmixing_coeffs = self.mixing_coeffs[0]
        
        #compute metropolis-hastings proposal
        log_helper = np.insert(
                np.cumsum(np.log(1 - self.betas)[:-1]), 0,0)
        inside_exp = np.dot(self.betas,
                            np.exp(log_helper))
        self.metropolis = inside_exp
        
        
    def initialise_parameters(self):
        """
        Initialises the parameter values of mu, tau using the hyperparameters.
        """
        
        self.precision[0] = np.random.gamma(
                self.hyparam['precision_shape'], 
                1.0/self.hyparam['precision_rate'])
        
        self.means[0] = np.random.normal(
                self.hyparam['means_loc'],
                np.sqrt(1.0/(self.precision[0]*self.hyparam['means_scale'])))

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
    
    def update_mixing_coeffs(self, it):
        """
        Samples beta using Metropolis-Hastings step conditional on the 
        updated auxiliary components.
        :param it: the MCMC iteration number.
        """
        component_sums = np.sum(self.auxiliary, axis=0)
        
        proposal_betas = self.betas.copy()
        for i in range(self.n_components):
            a = component_sums[i] + 1
            b = self.alpha + np.sum(component_sums[i+1:])
            proposal_betas[i] = np.random.beta(a, b + 0.0001)
        
            log_helper_p = np.insert(np.cumsum(np.log(1 - proposal_betas)[:-1]), 0, 0) + np.log(proposal_betas)
            log_helper_b = np.insert(np.cumsum(np.log(1 - self.betas)[:-1]), 0, 0) + np.log(self.betas)
            
            inside_p = np.delete(log_helper_p, np.arange(i))
            inside_b = np.delete(log_helper_b, np.arange(i))
            
            total = np.sum(np.exp(inside_b) - np.exp(inside_p))
            
            if np.random.rand() < np.exp(self.rate*self.T*total):
                self.betas = proposal_betas.copy()
                proposal_betas = self.betas.copy()

        self.currentmixing_coeffs = self.compute_stick_breaking_weights(
                self.betas)
            
        rate = 1 - np.sum(np.log(1-self.betas))
        shape = 1
        self.alpha = np.random.gamma(shape, 1/rate)
        """
            log_helper = np.log(1 - self.betas)
            sumlog = np.sum(log_helper[:i])
            proposal_top = (-self.n_segments*self.rate*self.delta*
                                  proposal_betas[i]*np.exp(sumlog))
            proposal_bottom = (-self.n_segments*self.rate*self.delta*
                               self.betas[i]*np.exp(sumlog))
            if np.random.rand() < np.exp(proposal_top - proposal_bottom):
                self.betas[i] = proposal_betas[i]
                print(proposal_top)
                print(proposal_bottom)
        
        self.currentmixing_coeffs = self.compute_stick_breaking_weights(
                self.betas)
        """
        """
        #compute metropolis-hastings proposal
        log_helper = np.insert(
                np.cumsum(np.log(1 - proposal_betas)[:-1]), 0,0)
        inside_exp = np.dot(proposal_betas,
                            np.exp(log_helper))
        proposal = inside_exp
        
        if np.random.rand() < np.exp(proposal - self.metropolis):
            self.metropolis = proposal
            self.betas = proposal_betas
            self.currentmixing_coeffs = self.compute_stick_breaking_weights(
                    self.betas)
        """
    
    def update_parameters(self, it):
        """
        Samples the parameter values condiitonal on the auxiliary variable
        using the conjugate prior formulas.
        :param it: the MCMC iteration number.
        """
        
        segment_sums = np.sum(self.auxiliary, axis=1)  
        Q, P, R = self.computeQPR(segment_sums)
        invP = np.linalg.inv(P)
        
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
        zerotrun_pois = (ZeroTruncatedPoission(rate=np.sum(self.currentmixing_coeffs)
        *self.delta).sample(self.n_segments))
        
        for seg in range(self.n_segments):
            prop = zerotrun_pois[seg]
            prop_components = np.random.multinomial(prop, 
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
    
    def propose_move(self):
        component_sums = np.sum(self.auxiliary, axis=0)
        component_weights = component_sums/np.sum(component_sums)
        indexes = np.random.choice(self.n_components, size=2)
        if indexes[0] != indexes[1]:
            self.propose_merge(indexes[0], indexes[1], component_weights)
        else:
            self.propose_split(indexes[0], component_weights)
    
    def propose_split(self, i, component_weights):
        pass
    
    def propose_merge(self, i, j, component_weights):
        target = j if np.argmax([component_weights[i], component_weights[j]]) else i
        source = i if target == j else j
        
        proposal_means = self.currentmeans.copy()
        proposal_means[target] = (component_weights[target]*proposal_means[target] + 
                      component_weights[source]*proposal_means[source])/(component_weights[target] + component_weights[source])
        proposal_means = np.delete(proposal_means, source)
        proposal_auxiliary = self.auxiliary.copy()
        proposal_auxiliary[:, target] += proposal_auxiliary[:, source]
        proposal_auxiliary = np.delete(proposal_auxiliary, source, axis=1)
        
        
        log_MH = 0
        for seg in range(self.n_segments):
            accept_top = Gaussian(np.dot(proposal_auxiliary[seg], proposal_means),
                                  np.sum(proposal_auxiliary[seg])/self.currentprecision).pdf(self.jumps[seg])
            accept_bot = Gaussian(
            np.dot(self.auxiliary[seg], self.currentmeans),
            np.sum(self.auxiliary[seg])/self.currentprecision).pdf(
                    self.jumps[seg])
            
            log_MH += np.log(accept_top) - np.log(accept_bot)
        print(log_MH)
        if np.random.rand() < np.exp(log_MH):
            self.merge(target, source, proposal_means)
    
    def merge(self, target, source, proposal_means):
        self.n_components -= 1
        self.currentmeans = proposal_means
        self.means = np.delete(self.means, source, axis=1)
        self.betas = np.delete(self.betas, source)
        proposal_auxiliary = self.auxiliary
        proposal_auxiliary[:, target] += proposal_auxiliary[:, source]
        proposal_auxiliary = np.delete(proposal_auxiliary, source, axis=1)
        self.auxiliary = proposal_auxiliary
        proposal_mixing_coeffs = self.mixing_coeffs
        proposal_mixing_coeffs[:, target] += proposal_mixing_coeffs[:, source]
        proposal_mixing_coeffs = np.delete(proposal_mixing_coeffs, source, axis=1)
        self.mixing_coeffs = proposal_mixing_coeffs
        self.hyparam['means_loc'] = np.zeros(self.n_components)
        print("ok")
        
        
        
                    
                    
            
            
