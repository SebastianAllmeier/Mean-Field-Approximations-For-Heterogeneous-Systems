"""
This class contains code to simulate or compute the refined mean field
approximation for a linear cache model. 
"""
import os
import numpy as np
from rmf.costcachingmodel import cache_linear_zipf
from exact.cache_exact import proba_all_item_zipf

# add functionality for windows operating system
import platform

# We first make sure that the necessary scripts are there:
os.system('cd simulation && make')
if not platform.system() == "Windows":
    os.system('if [ ! -e data ]; then mkdir data/; fi')

class CacheModel:
    """
    This class contains ...
    """
    def __init__(self, N, M, alpha):
        """
        Initialize the code
        """
        self.N = N
        self.M = M
        self.alpha = alpha
        p = 1/(1+np.arange(0,N))**alpha
        self.populatities = p/sum(p)

    def output_name(self):
        m_string_output = '-'.join([str(mi) for mi in self.M])
        if platform.system() == "Windows":
            return 'data\\simu-N{}-M{}-a{}'.format(self.N, m_string_output, self.alpha)
        else:
            return './data/simu-N{}-M{}-a{}'.format(self.N, m_string_output, self.alpha)

    def exact(self, force_recompute=False):
        """
        Exact steady-state distribution, computed by using the product form.

        Output:
        - a vector x of size N * len(M), where x[i,j] is the steady-state probability
        that item 'i' is in list 'j'.
        """
        output_name = self.output_name()+'-exact.txt'
        if not os.path.exists(output_name) or force_recompute:
            np.savetxt(output_name, proba_all_item_zipf(self.N, self.M, self.alpha))
        return np.loadtxt(output_name)

    def simulate(self, seed=None, force_recompute=False):
        """
        Estimation of the steady-state distribution by simulation.

        Input:
        - seed (default=None): seed of the random generator used in the simulation
        - force_recompute (default=False): does not attempt to load the result

        Output:
        - a vector x of size N * len(M), where x[i,j] is the average time that item 'i' was in list 'j'.
        """
        if seed is None:
            seed = np.random.randint(10000000)
        m_string = ' '.join([str(mi) for mi in self.M])
        output_name = self.output_name()+'-s'+str(seed)+'.txt'
        if platform.system() == "Windows":
            command = 'simulation\\simu.exe N{} M{} a{} s{} H'.format(self.N, m_string, self.alpha, seed)
        else:
            command = './simulation/simu N{} M{} a{} s{} H'.format(self.N, m_string, self.alpha, seed)
        if not os.path.exists(output_name) or os.stat(output_name).st_size == 0 or force_recompute:
            if platform.system() == "Windows":
                os.system(command + ' > data\\tmp')
                os.rename('data/tmp', output_name)
            else:
                os.system(command+' > ./data/tmp')
                os.system('mv ./data/tmp '+output_name)
        return np.loadtxt(output_name)

    def many_simulations(self, number_of_simulations):
        """
        Estimation of steady-state distribution + confidence interval.

        Output: (x, c), where:
        - x is of size N*len(M)
        - error1[i,j] is the absolute error of x[i,j]    (of one simulation vs the mean)
        - error2[i,j] is the mean square error of x[i,j] (of one simulation vs the mean)
        """
        mean = np.zeros((self.N, len(self.M)))
        error1 = np.zeros((self.N, len(self.M)))
        error2 = np.zeros((self.N, len(self.M)))
        for seed in range(number_of_simulations):
            mean += self.simulate(seed) / number_of_simulations
        for seed in range(number_of_simulations):
            error1 += np.abs(self.simulate(seed)-mean) / (number_of_simulations-1)
            error2 += (self.simulate(seed)-mean)**2 / (number_of_simulations-1)
        return mean, error1, error2

    def mean_field(self, force_recompute=False):
        """
        Estimation of the steady-state distribution by mean field approximation.

        Input:
        - force_recompute (default=False): does not attempt to load the result

        Output:
        - a vector x of size N * len(M), where x[i,j] is the average time that item 'i' was in list 'j'.
        """
        output_name = self.output_name()+'-mf.txt'
        if not os.path.exists(output_name) or force_recompute:
            model = cache_linear_zipf(self.M, self.N, self.alpha)
            np.savetxt(output_name, model.fixed_point2D()[:,1:])
        return np.loadtxt(output_name)

    def refined_mean_field(self, force_recompute=False):
        """
        Estimation of the steady-state distribution by refined mean field approximation.

        Input:
        - force_recompute (default=False): does not attempt to load the result

        Output:
        - a vector x of size N * len(M), where x[i,j] is the average time that item 'i' was in list 'j'.
        """
        output_name = self.output_name()+'-rmf.txt'
        if not os.path.exists(output_name) or force_recompute:
            model = cache_linear_zipf(self.M, self.N, self.alpha)
            pi, V, W = model.refinedMF_steadyState()
            np.savetxt(output_name, (pi+V)[:,1:])
        return np.loadtxt(output_name)
    
    def total_error_mean_field(self, aggregate=False):
        if not aggregate:
            mf = self.mean_field()
            exact = self.exact()
        else:
            mf = self.populatities@self.mean_field()
            exact = self.populatities@self.exact()
        return np.sum(np.abs(mf-exact))

    def total_error_refined_mean_field(self, number_of_simulations=1, aggregate=False):
        if not aggregate:
            rmf = self.refined_mean_field()
            exact = self.exact()
        else:
            rmf = self.populatities@self.refined_mean_field()
            exact = self.populatities@self.exact()
        return np.sum(np.abs(rmf-exact))

    def total_error_simulation(self, number_of_simulations=1, aggregate=False):
        if not aggregate:
            simu, _, _ = self.many_simulations(number_of_simulations)
            exact = self.exact()
        else:
            simu, _, _ = self.many_simulations(number_of_simulations)
            simu = self.populatities@simu
            exact = self.populatities@self.exact()
        return np.sum(np.abs(simu-exact))
        
