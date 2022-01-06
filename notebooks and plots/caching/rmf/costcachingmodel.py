"""
This library contains what is needed to define the model of cache of [INSERT CITATION]
and the code to computes a "mean-field" (FPI of [ref]) and "refined mean field" (RMF of [ref])
"""
import numpy as np
import time
from numpy import tensordot as tsdot
import scipy.optimize
import scipy.integrate as integrate
from scipy.linalg import solve_continuous_lyapunov, inv
from itertools import permutations
from scipy.special import factorial

import sys, os
PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)
sys.path.append(DIR_PATH)
print(DIR_PATH)
import define_derivatives
from rmftool.refinedRefined_fixedPoint import computePiV, computeW


class CacheModel():
    """
    Class
    """
    def __init__(self, m, rates=None):
        """
        - m is a list of "list sizes" (m[0] is the size of the "outside")
        - rates[i,l,lp] is the rate at which an item is moved from list l to list lp

        We should have rates.shape[0] == sum(m)
        """
        self.m = np.array(m)
        self.h = len(m)
        if rates is not None:
            assert len(rates) == sum(m), "Dimension error: list lengths should sum to the number of items"
            self.rates = np.array(rates)
            self.n = rates.shape[0]
            self.update_predecessors_successors()
        else:
            self.predecessors = None
            self.successors = None

    def update_predecessors_successors(self):
        """
        This function defines a predecessor[i,l] = list from which j can came (should be unique!)
        """
        self.predecessors = -np.ones((self.n,self.h),dtype=int)
        self.successors = [[] for i in range(self.h)]
        for i in range(self.n):
            for l in range(self.h):
                for lp in range(self.h):
                    if l != lp and self.rates[i, l, lp] > 0:
                        assert self.predecessors[i, lp] == -1, 'The "rates" are not a tree'
                        self.predecessors[i, lp] = l
                        if lp not in self.successors[l]:
                            self.successors[l].append(lp)

    def compute_gamma(self, i, l):
        """Recursive definition of gamma"""
        if l == 0 or self.predecessors[i, l] == -1:
            return 1
        pred = self.predecessors[i, l]
        if pred >= 0:
            return self.rates[i, pred, l]*self.compute_gamma(i, pred)
        return 0

    def fixed_point2D(self, verbose=False):
        """
        Computes the fixed point of the model (as a 2D vector of size (n,h)
        """
        self.update_predecessors_successors()
        # We first update the values of gamma
        gamma = [[self.compute_gamma(i, l) for l in range(self.h)] for i in range(self.n)]
        # We are now ready for iterations
        xi = np.ones(self.h)
        iterations = 0
        while iterations == 0 or (iterations < 1000 and sum(abs(xi_old-xi)>=1e-8)):
            xi_old = np.copy(xi)
            for l in range(1,self.h):
                #a = [sum([gamma[i][lp]*xi_old[lp]/gamma[i][l] for lp in range(self.h) if l != lp]) for i in range(self.n)]
                a1 = [gamma[i][l] for i in range(self.n)]
                b1 = [sum([gamma[i][lp]*xi_old[lp] for lp in range(self.h) if l != lp]) for i in range(self.n)]
                #xi[l] = 1/fixed_point_function(a, self.m[l])
                newx = fixed_point_function2(a1, b1, self.m[l])
                xi[l] = newx #1/fixed_point_function(a, self.m[l])
                #print('l=',l,'xi=',xi[l], 'newx=',newx)
            if verbose:
                print("\riteration=", iterations, sum(abs(xi-xi_old)), end='         ')
                print(xi-xi_old)
            iterations += 1
        pi = np.zeros((self.n, self.h))
        for i in range(self.n):
            for l in range(self.h):
                pi[i, l] = gamma[i][l]*xi[l]
            pi[i, :] /= sum(pi[i, :])
        return pi

    def old_fixed_point2D(self, verbose=False):
        """
        Computes the fixed point of the model (as a 2D vector of size (n,h)
        """
        self.update_predecessors_successors()
        # We first update the values of gamma
        gamma = [[self.compute_gamma(i, l) for l in range(self.h)] for i in range(self.n)]
        # We are now ready for iterations
        xi = np.ones(self.h)
        for _ in range(1000): # 20 iterations seems sufficient
            for l in range(1, self.h):
                a = [sum([gamma[i][lp]*xi[lp]/gamma[i][l] for lp in range(self.h) if l != lp]) for i in range(self.n)]
                xi[l] = 1/fixed_point_function(a, self.m[l])
            if verbose:
                print("\riteration=", iterations, sum(abs(xi-xi_old)), end='         ')

        pi = np.zeros((self.n, self.h))
        for i in range(self.n):
            for l in range(self.h):
                pi[i, l] = gamma[i][l]*xi[l]
            pi[i, :] /= sum(pi[i, :])
        return pi

    def pure_FPI(self, verbose=False):
        self.update_predecessors_successors()
        # We first update the values of gamma
        gamma = [[self.compute_gamma(i, l) for l in range(self.h)] for i in range(self.n)]
        # We are now ready for iterations
        xi = np.ones(self.h)
        pi = np.ones((self.n, self.h))/self.h
        for iterations in range(1000):
            xi_old = np.copy(xi)
            for l in range(1, self.h):
                xi[l] = self.m[l]/np.sum([gamma[k][l]*(1-np.sum(pi[k][1:])) for k in range(self.n)])
            for i in range(self.n):
                for l in range(self.h):
                    pi[i, l] = gamma[i][l]*xi[l]
                pi[i, :] /= sum(pi[i, :])
            if verbose:
                print("\riteration=", iterations,sum(abs(xi-xi_old)),end='         ')
                print(xi-xi_old)
            #print(np.sum(pi, 0))
        return pi

    def fixed_point(self):
        """
        Computes the fixed point of the model (as a 1D vector of size n*h)
        """
        return self.fixed_point2D().reshape((self.n*self.h))

    def index(self, i, l):
        """
        Makes the correspondance between 2D indices and 1D indices
        - index(i,l) is the 1D index of "item i" in "list l"
        """
        return l + i * self.h

    def compute_fp(self, x):
        """
        Compute the first derivative of the drift (x is a 1D vector)
        """
        dF = np.zeros((self.n,self.h,self.n,self.h)) # First derivative
        x_2D = x.reshape((self.n,self.h))
        for i in range(self.n):
            for k in range(self.n):
                for l in range(self.h):
                    for lp in self.successors[l]: # We exchange i in 'l' with k in 'lp'$
                        rateDerivativeI = self.rates[i,l,lp]*x_2D[k, lp]/self.m[lp] # derivative wrt x_{i,:}
                        rateDerivativeK = self.rates[i,l,lp]*x_2D[i, l]/self.m[lp]  # derivative wrt x_{k,:}
                        for (sign, idx0, idx1) in [(-1,i,l), (1,i,lp), (1,k,l), (-1,k,lp)]:
                            dF[idx0, idx1, i, l] += sign*rateDerivativeI
                            dF[idx0, idx1, k, lp] += sign*rateDerivativeK
        return dF.reshape((self.n*self.h,self.n*self.h))

    def compute_fpp(self, x):
        """
        Compute the second derivative of the drift at x (x is a 1D vector)
        """
        ddF = np.zeros((self.n,self.h,self.n,self.h,self.n,self.h)) # Second derivative
        for i in range(self.n):
            for k in range(self.n):
                for l in range(self.h):
                    for lp in self.successors[l]:
                        for (sign,idx0,idx1) in [(-1,i,l),(1,i,lp),(1,k,l),(-1,k,lp)]:
                            ddF[idx0,idx1,i,l,k,lp] += sign*self.rates[i,l,lp]/self.m[lp]
                            ddF[idx0,idx1,k,lp,i,l] += sign*self.rates[i,l,lp]/self.m[lp]
        return ddF.reshape((self.n*self.h,self.n*self.h,self.n*self.h))

    def compute_q(self, x):
        Q = np.zeros( (self.n,self.h,self.n,self.h) ) # Co-variance Matrix Q
        x_2D = x.reshape((self.n,self.h))
        for i in range(self.n):
            for k in range(self.n):
                if i!=k:
                    for l in range(self.h):
                        for lp in self.successors[l]:
                            rateTransition = self.rates[i,l,lp] * x_2D[i,l] * x_2D[k,lp]/self.m[lp]
                            ## NOT DONE BELOW 
                            for (sign0,idx00,idx01) in [(-1,i,l),(1,i,lp),(1,k,l),(-1,k,lp)]:
                                for (sign1,idx10,idx11) in [(-1,i,l),(1,i,lp),(1,k,l),(-1,k,lp)]:
                                    Q[idx00,idx01,idx10,idx11] += sign0*sign1*rateTransition
        return Q.reshape((self.n*self.h,self.n*self.h))

    # Numba versions
    def compute_fp(self, x):
        return define_derivatives.compute_fp(self.rates, self.m, x)
    def compute_fpp(self, x):
        return define_derivatives.compute_fpp(self.rates, self.m, x)
    def compute_q(self, x):
        return define_derivatives.compute_q(self.rates, self.m, x)

    def compute_matrices_to_reduce_dim(self,Fp):
        """
        """
        rank_of_jacobian = np.linalg.matrix_rank(Fp)
        C = np.zeros((self.n*self.h,self.n*self.h))
        d=0
        for l in range(self.h-1):
            for i in range(self.n-1):
                C[d, self.index(i,l)] = 1
                d += 1
        t=time.time()
        U,s,V = scipy.linalg.svd(Fp)
        C[rank_of_jacobian:,:] = U.transpose()[rank_of_jacobian:,:]
        Cinv = scipy.linalg.inv(C)
        return C, Cinv, rank_of_jacobian
    
    def reduce_dimension_fpq(self, Fp, Q):
        P,Pinv, rank=self.compute_matrices_to_reduce_dim(Fp)
        Fp = (P@Fp@Pinv)[0:rank,0:rank]
        Q = (P@Q@P.transpose())[0:rank,0:rank]
        return Fp, Q, P, Pinv, rank

    def reduce_dimension(self,Fp,Fpp,Q):
        """
        Reduce dimension (to obtain an invertible matrix)
        """
        P, Pinv, rank=self.compute_matrices_to_reduce_dim(Fp)
        Fp = (P@Fp@Pinv)[0:rank,0:rank]
        Fpp = tsdot(tsdot(tsdot(P,Fpp,1), Pinv,1),Pinv,axes=[[1],[0]])
        Fpp = Fpp[0:rank,0:rank,0:rank]
        Q = (P@Q@P.transpose())[0:rank,0:rank]
        return(Fp,Fpp,Q, P, Pinv,rank)

    def refinedMF_steadyState(self, verbose=False):
        t = time.time()
        pi = self.fixed_point().reshape((self.n*self.h))
        if verbose: print("Time to compute fixed point:",time.time()-t); t=time.time()
        Fp = self.compute_fp(pi)
        if verbose: print("  Time for FP:",time.time()-t); t=time.time()
        #Fpp = self.compute_fpp(pi)
        #if verbose: print("  Time for Fpp:",time.time()-t); t=time.time()
        Q = self.compute_q(pi)
        if verbose: print("  Time for Q:",time.time()-t); t=time.time()
        V, W = self.compute_vw_test(Fp, Q, verbose)
        if verbose: print("Time to compute RMF:",time.time()-t); t=time.time()
        #print('difference W=', np.sum( np.abs(W-Wnew)), 'difference V=', np.sum( np.abs(Vnew-V)), np.sum( np.abs(Vnew)), np.sum( np.abs(V)))
        return pi.reshape(self.n,self.h),V.reshape(self.n,self.h),W.reshape(self.n,self.h,self.n,self.h)

    def compute_vw_test(self, Fp, Q, verbose=False):
        t = time.time()
        Fp, Q, P, Pinv, rank = self.reduce_dimension_fpq(Fp,Q)
        if verbose: print("    -> Time to reduce dim:",time.time()-t); t=time.time()
        W = computeW(Fp,Q)
        if verbose: print("    -> Time to W (Lyapynov eq):",time.time()-t); t=time.time()
        W = Pinv[:,0:rank]@W@Pinv.transpose()[0:rank,:]
        if verbose: print("    -> Time to expand:",time.time()-t); t=time.time()
        tmp = self.tensordot_by_fpp(W/2)
        if verbose: print("    -> Time to mulitply Fpp and W:",time.time()-t); t=time.time()
        FppW = (P@tmp)[0:rank]
        if verbose: print("    -> Time to re-mulitply by P:",time.time()-t); t=time.time()
        V =  Pinv[:,0:rank]@(-np.tensordot(inv(Fp),FppW, 1) )
        if verbose: print("    -> Time for V:",time.time()-t); t=time.time()
        return V,W

    def tensordot_by_fpp(self, W, debug=False):
        """
        Test (to be verified later)
        """
        is_successor = define_derivatives.successor(self.rates)
        res = define_derivatives.tensordot_by_fpp(W, is_successor, self.rates, self.m, self.h)
        if debug:
            pi = self.fixed_point().reshape((self.n*self.h))
            Fpp = self.compute_fpp(pi)
            test = np.tensordot(Fpp, W, 2)
            return test, res
        return res

    def compute_vw(self, Fp, Fpp, Q, verbose=False):
        t = time.time()
        Fp, Q, P, Pinv, rank = self.reduce_dimension_fpq(Fp,Q)
        if verbose: print("    -> Time to reduce dim:",time.time()-t); t=time.time()
        W = computeW(Fp,Q)
        if verbose: print("    -> Time to W (Lyapynov eq):",time.time()-t); t=time.time()
        W = Pinv[:,0:rank]@W@Pinv.transpose()[0:rank,:]
        if verbose: print("    -> Time to expand:",time.time()-t); t=time.time()
        tmp = np.tensordot(Fpp,W/2,2)
        if verbose: print("    -> Time to mulitply Fpp and W:",time.time()-t); t=time.time()
        FppW = (P@tmp)[0:rank]
        if verbose: print("    -> Time to re-mulitply by P:",time.time()-t); t=time.time()
        V =  Pinv[:,0:rank]@(-np.tensordot(inv(Fp),FppW, 1) )
        if verbose: print("    -> Time for V:",time.time()-t); t=time.time()
        return V,W

    def initial_value(self):
        initial2D = np.zeros((self.n,self.h))
        number_in_list_l = 0
        l = 0
        for i in range(self.n):
            if number_in_list_l == self.m[l]:
                l+=1
                number_in_list_l = 0
            initial2D[i,l] = 1
        return initial2D.reshape((self.n*self.h))


    def transient_trajectory(self, Tmax = 1000):
        """
        Initial conditions: 
        - the first m[0] items are outside the cache
        - the next m[1] items are in list "1", etc. 
        """
        y0 = self.initial_value()
        def f(t, y):
            return define_derivatives.drift(self.rates, self.m, y)
        t_eval = np.linspace(0,Tmax,10000)
        sol_ode = scipy.integrate.solve_ivp(f, t_span=[0,Tmax], y0 = y0, t_eval=t_eval)
        y2D_sol = sol_ode.y.reshape((self.n,self.h,len(t_eval)))
        popularities = self.rates[:,0,1]
        miss_function_time = np.tensordot(popularities, y2D_sol[:,0,:], 1)
        pi = self.fixed_point2D()
        steady_miss = np.sum(pi[:,0]* popularities )
        return t_eval, miss_function_time, np.ones(len(t_eval))*steady_miss

    def drift_r_vector(self, X):
        dim = self.n*self.h
        x = X[0:dim]
        V = X[dim:2*dim]
        W = X[2*dim:].reshape((dim,dim))
        F   = define_derivatives.drift(self.rates, self.m, x)
        Fp  = define_derivatives.compute_fp(self.rates, self.m, x)
        Fpp = define_derivatives.compute_fpp(self.rates, self.m, x)
        Q = define_derivatives.compute_q(self.rates, self.m, x)
    
        dV = np.tensordot(Fp,V,1) + np.tensordot(Fpp,W,2)/2
        dW = 2*symetric_tensor(np.tensordot(Fp,W,1))+Q
        dX = np.zeros(2*dim+dim**2)
        dX[0:dim] = F
        dX[dim:2*dim] = dV
        dX[2*dim:] = dW.reshape(dim**2)
        return(dX)

    def refinedMF_transient_manual(self,Tmax=100,nbSteps=1000):
        T=np.linspace(0,Tmax,nbSteps)
        XVW_0 = np.zeros((2*self.n*self.h+(self.n*self.h)**2))
        XVW_0[0:self.n*self.h] = self.initial_value()
        XVW = np.zeros((2*self.n*self.h+(self.n*self.h)**2, nbSteps))
        T = np.linspace(0,Tmax,nbSteps)
        for t in range(0,nbSteps):
            XVW[:,t] = XVW_0
            XVW_0 = XVW_0 + self.drift_r_vector(XVW_0)*Tmax/nbSteps
        X = XVW[0:self.n*self.h,:].reshape(self.n, self.h,nbSteps)
        V = XVW[self.n*self.h:2*self.n*self.h,:].reshape(self.n,self.h,nbSteps)
        W = XVW[2*self.n*self.h:,:]
        popularities = self.rates[:,0,1]
        miss_mf = np.tensordot(popularities, X[:,0,:], 1)
        miss_rmf = np.tensordot(popularities, V[:,0,:], 1)
        return T,X,V,W, miss_mf, miss_rmf
    

    def refinedMF_transient(self,Tmax=100,nbSteps=100):
        T=np.linspace(0,Tmax,nbSteps)
        XVW_0 = np.zeros((2*self.n*self.h+(self.n*self.h)**2))
        XVW_0[0:self.n*self.h] = self.set_initial_value()
        numericalInteg = integrate.solve_ivp(
            lambda t,x : self.drift_r_vector(x),
            [0,Tmax], XVW_0,t_eval=T,rtol=1e-6)
        X = numericalInteg.y[0:self.n*self.h,:].reshape(self.h,self.n,nbSteps)
        V = numericalInteg.y[self.n*self.h:2*self.n*self.h,:].reshape(self.h,self.n,nbSteps)
        W = numericalInteg.y[2*self.n*self.h:,:]
        return T,X,V,W

def fixed_point_function(a, m):
    """
    Solves an equation of the form sum_i 1/(1+a_i z) = m with m < len(a)
    """
    a = np.array([x for x in a if x<np.inf])
    def f(z):
        return np.sum(1/(1+a*z))-m
    zmin, zmax = 0, 1
    while f(zmax) > 0:
        zmin, zmax = zmax, 2*zmax
    return scipy.optimize.brentq(f, zmin, zmax)
    

def fixed_point_function2(a, b, m):
    """
    Solves an equation of the form sum_i a_i z /(b_i+a_i z) = m with m < len(a)
    """
    b_over_a = np.array(b)/np.array(a)
    def f(z):
        return np.sum(z/(b_over_a+z))-m
    zmin, zmax = 0, 1
    while f(zmax) < 0:
        zmin, zmax = zmax, 2*zmax
    return scipy.optimize.brentq(f, zmin, zmax)


def cache_linear_zipf(m, n, alpha):
    """
    Returns a linear model from a zipf distribution
    """
    p = 1/(1+np.arange(0,n))**alpha
    p = p/sum(p)
    return cache_linear(m, p)

def cache_linear(m, p):
    """
    Returns a linear model from probability vector
    """
    n = len(p)
    new_m = np.zeros(len(m)+1, dtype=int)
    new_m[1:] = m
    new_m[0] = n-sum(m)
    h = len(new_m)
    rates = np.zeros((n,h,h))
    for i in range(n):
        for l in range(h-1):
            rates[i,l,l+1] = p[i]
    return CacheModel(new_m,rates)

def symetric_tensor(T):
    n = len(T.shape)
    newT = np.zeros(shape=(T.shape),dtype=np.float32)
    for sigma in permutations(range(n)):
        newT += np.transpose(T,axes=list(sigma))
    return(newT/factorial(n))

