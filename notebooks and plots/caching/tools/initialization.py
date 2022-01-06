# classes and function to simplify intialization
### classes are for sir model


import numpy as np

def initialization_of_parameters(nr_items, allow_R_to_S=True, seed=None):
    """

    Initialization of parameters where transitions from removed to susceptible
    are not possible.

    :param seed: random seed to reproduce results
    :param nr_items: nr of items of the sir model for which to generate the parameteres
    :return: alpha, beta, gamma, delta - np arrays of size
    """
    if seed is not None:
        np.random.seed(seed)
    if allow_R_to_S:
        alpha = Distributions.uniform(0., 2., nr_items)
        beta = Distributions.uniform(8., 15., nr_items)
        gamma = Distributions.uniform(5., 8., nr_items)
        delta = Distributions.uniform(3., 6., nr_items)
    if not allow_R_to_S:
        alpha = Distributions.uniform(0., 2., nr_items)
        beta = Distributions.uniform(5., 10., nr_items)
        gamma = Distributions.uniform(5., 6., nr_items)
        delta = np.zeros(shape=(nr_items))
    return alpha, beta, gamma, delta

class Distributions():
    """
    class basically wrapping numpy functions for better readability of the code
    """
    @staticmethod
    def uniform(a, b, n):
        return np.random.uniform(low=a, high=b, size=n)

    @staticmethod
    def gaussian(center, variation, n):
        return np.random.normal(loc=center, scale=variation, size=n)

class Transitions():
    """
    class specifying the transitions of the SIR model

    k       - generally refers to the k-th item (starting with 0)
    i       - index in a matrix or tensor (generally a tuple or list)
    size    - the size of the dynamic system (tuple or list)
    """

    def __init__(self, size=None):
        self.size = size

    # static method creating unit vectors
    @staticmethod
    def unit_vector(i, size):
        # i, size can be integers or lists / tuples of integers
        unit_i = np.zeros(size)
        unit_i[i] = 1
        return unit_i

    @staticmethod
    def S_to_I(k, size):
        # return transition vector from susceptible to infected
        return - Transitions.unit_vector((k, 0), size) + Transitions.unit_vector((k, 1), size)

    @staticmethod
    def rate_S_to_I(k, alpha, beta, x):
        """
        calculate infection rate by (\alpha_i + \beta_i * 1/n * \sum_k X_{k,I}) * X_{i,S}
        :param k: corresponding item
        :param alpha: independent infections
        :param beta: infection rate depending on the infected people
        :param x: a vector/matrix of the same size as the dynamical system
        :return: infection rate of item k (float)
        """
        return (alpha + beta * (1/x.shape[0]) * np.sum(x[:, 1])) * x[k, 0]

    @staticmethod
    def I_to_R(k, size):
        # return transition vector from infected to recovered / removed
        return - Transitions.unit_vector((k, 1), size) + Transitions.unit_vector((k, 2), size)

    @staticmethod
    def rate_I_to_R(k, gamma, x):
        return gamma * x[k, 1]

    @staticmethod
    def R_to_S(k, size):
        # return transition vector from recovered / removed to susceptible
        return - Transitions.unit_vector((k, 2), size) + Transitions.unit_vector((k, 0), size)

    @staticmethod
    def rate_R_to_S(k, delta, x):
        return delta * x[k, 2]