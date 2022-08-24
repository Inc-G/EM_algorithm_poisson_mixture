import numpy as np
import scipy

import pandas as pd
from matplotlib import pyplot as plt


class PoissonMixture():
    def __init__(self, K):
        """
        K: int. The number of components
        """
        self.K = K
        self.lambdas = []
        self.probs = []
        self.log_lik = []
            
    
    def compute_Poisson(self, X, lambdas):
        """
        lambdas: np.array of length K=number of classes.
        X: np.array of length N=number of datapts (X is the array of datapoints).
    
        returns: np.ndarray with shape (N,K) that in position [i,l] is the Poisson distribution with
        parameter lambda[l] and value x[i]
    
        - Construct an NxK matrix (concat_lambdas) with entry i, l equal to lambda_l^{x_i}
        - Replace concat_lambdas[i, l] with (concat_lambdas[i, l]*exp(-lambda[l]))/(x[i]!)
        """
        N = X.shape[0]
    
        concat_lambdas = np.concatenate([np.array(lambdas)[np.newaxis,...] for _ in range(N)], axis=0)
        concat_lambdas = np.transpose(np.transpose(concat_lambdas) ** X)
        numerator = concat_lambdas * np.exp(-lambdas)
        return np.transpose(np.transpose(numerator)/scipy.special.factorial(X))
    
    
    def expected_log_likelyhood(self, poissons, X, probs):
        """
        poissons: np.ndarray with shape (N,K). poissons[i,j] = Poisson(x[i], lambda[l]) (see self.compute_Poisson)
        X: np.array of length N=number of datapts (X is the array of datapoints).
        probs: np.array of non-negative floats that sum to 1.
        
        Returns the mean of log(p(X[i])) 
        """
        return np.mean(np.log(poissons@probs)) # poissons@probs is p(X)
    
    
    def _E_step(self,poissons, probs):
        """
        poissons: np.ndarray with shape (N,K). poissons[i,j] = Poisson(x[i], lambda[l]) (see self.compute_Poisson)
        probs: np.array of non-negative floats that sum to 1
    
        returns: np.ndarray with shape (N,K) that is p(Z|X)
    
        Computes the distribution q=p(Z|X) as the normalization of poissons * probs
        """
        unnormalized_q = poissons * probs #multiplies column l of poissons by probs[l]
        q = np.transpose(np.transpose(unnormalized_q)/np.sum(unnormalized_q, axis=1)) #each row sums to 1
        return q
    

    def _M_step_probs(self,q):
        """
        q: np.ndarray with shape (N,K) representing p(Z|X).
        
        Performs the M step for computing the probabilities of the components of the mixture of poissons
        """
        return np.sum(q, axis=0)/np.sum(q)
    

    def _M_step_lambdas(self,q, X):
        """
        q: np.ndarray with shape (N,K) representing p(Z|X).
        X: np.array (X is the array of datapoints).
        
        Performs the M step for computing the parameters for each component of the mixture of poissons
        """
        return ((X[np.newaxis, ...]@q)/np.sum(q, axis=0))[0]
    
    
    def fit(self, X, max_iterations=1000, epsilon=0.001, max_lambda=5, min_lambda=.2,
           plot_results=False, plot_log_lik=False, plot_probs=True, plot_lambdas=True):
        """
        X: np.array. The datapoints
        max_iterations: int
        epsilon: float
        max_lambda: float
        min_lambda: float
        plot_results: bool. Same as plot_log_lik, plot_probs, plot_lambdas.
        
        
        returns: None
        updates: self.lambdas, self.probs and self.log_lik with the sequence of lambdas, probs and log_lik obtained
        while running the EM algorithm
        
        
        At the beginning of the EM algorithm the lambdas are  uniformly sampled from [min_lambda, max_lambda]
        
        Runs the EM algorithm for either max_iterations or until: 
            abs(expected_log_likelyhood[step i] - expected_log_likelyhood[step i+1]) < epsilon
        If plot_results, it plots the results using self.plot_results. 
        """
        ## Initialize the probabilities and the lambdas
        unnormalized_probs = np.random.rand(self.K)
        probs = unnormalized_probs/unnormalized_probs.sum()
        
        ## Initialize the lambdas
        lambdas = min_lambda + max_lambda*np.random.rand(self.K)
        
        loglik = - np.inf
        
        past_expected_log_likelihoods = []
        past_lambdas = []
        past_probs = []
        
        for _ in range(max_iterations):
            poissons = self.compute_Poisson(X, lambdas)
            
            current_log_l = self.expected_log_likelyhood(poissons, X, probs)
            if loglik > epsilon + current_log_l:
                break
                        
            loglik = current_log_l
            
            q = self._E_step(poissons, probs)
            
            probs = self._M_step_probs(q)
            lambdas = self._M_step_lambdas(q, X)
            
            past_expected_log_likelihoods.append(current_log_l)
            past_probs.append(probs)
            past_lambdas.append(lambdas)
            
        self.lambdas = past_lambdas
        self.probs = past_probs
        self.log_lik = past_expected_log_likelihoods
        
        if plot_results:
            self.plot_results(plot_log_lik, plot_probs, plot_lambdas)
            
    
    def plot_results(self, plot_log_lik=False, plot_probs=True, plot_lambdas=True):
        """
        plots the results of EM algorithm. If plot_log_lik it plots the log likelyhoods, if plot_probs
        it plots the probabilities of each component, if plot_lambdas it plots the lambdas of each cluster.
        """
        if plot_lambdas:
            pd.DataFrame(np.array(self.lambdas)).plot(title='Lambdas',
                                                      figsize=(10,8),
                                                      xlabel='Steps EM algorithm',
                                                      ylabel='lambdas')
            plt.show()
            
        if plot_probs:
            pd.DataFrame(np.array(self.probs)).plot(title='Probability components',
                                                    figsize=(10,8),
                                                    xlabel='Steps EM algorithm',
                                                    ylabel='probability components')
            plt.show()
        
        if plot_log_lik:
            pd.DataFrame(np.array(self.log_lik)).plot(title='Log likelyhood',
                                                      figsize=(10,8),
                                                      xlabel='Steps EM algorithm',
                                                      ylabel='log likelyhood')
            plt.show()