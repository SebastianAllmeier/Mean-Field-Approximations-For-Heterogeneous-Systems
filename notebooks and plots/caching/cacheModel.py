import rmftool as rmf
import numpy as np
import matplotlib.pylab as plt
import time as ti
import scipy.integrate

from rmf_tool.src.refinedRefined_transientRegime import drift_r_vector, drift_rr_vector
from rmf_tool.src.refinedRefined_fixedPoint import computePiV,computePiVA


import random as rnd


def zipf(n, alpha):
    p = np.array([(i + 1) ** (-alpha) for i in range(n)])
    return p / np.sum(p)


class cacheRANDmDDPP(rmf.DDPP):
    def __init__(self, p, m):
        """

        :param p: - item distribution
        :param m: - cache sizes
        """
        super(cacheRANDmDDPP, self).__init__()

        self.number_of_lists = len(m)
        self.number_of_items = len(p)
        self.model_dimension = self.number_of_items * (self.number_of_lists + 1)
        self.m = m
        self.p = p
        nr_items = len(p)  # number of items
        nr_lists = len(m)  # number of lists

        for i in range(nr_items):
            for j in range(nr_items):
                if i != j:
                    for k in range(nr_lists):
                        ell = np.zeros(self.model_dimension)
                        ell[self.index(i, k)] += -1
                        ell[self.index(i, k + 1)] += 1
                        ell[self.index(j, k + 1)] += -1
                        ell[self.index(j, k)] += 1
                        rate = 'lambda x:{}*x[{}]*x[{}]/{}'.format(self.p[i], self.index(i, k), self.index(j, k + 1),
                                                                   m[k])
                        self.add_transition(ell, eval(rate))

        # we now define the initial state with for caches that have a size m[0]...m[h]
        initial_state = np.zeros(self.model_dimension)
        # we put the first k element in list 1, the next in list 2, etc.
        object_index = 0
        initial_obj = np.arange(nr_items)
        # np.random.shuffle(initial_obj)

        for _list in range(nr_lists-1, -1, -1):
            for i in range(m[_list]):
                initial_state[self.index(initial_obj[object_index], _list + 1)] = 1
                object_index += 1
        # initialize non cached items
        for i in range(nr_items - sum(m)):
            initial_state[self.index(initial_obj[object_index], 0)] = 1
            object_index += 1
        self.set_initial_state(initial_state)


    def index(self, i, k):
        return (i + k * self.number_of_items)

    def fixed_point(self):
        drift = self.defineDrift()
        T = np.linspace(0, 10000, 100)
        X = scipy.integrate.odeint(lambda x, t: drift(x), self._x0, T)
        return (X[-1, :])

    def hit_rate(self, x, list_number):
        return np.sum([self.p[i] * x[self.index(i, list_number)] for i in range(self.number_of_items)])

    def hit_rates1d(self, vector):
        if len(vector.shape) == 1:
            return np.array([self.hit_rate(vector, i) for i in range(self.number_of_lists + 1)])
        else:
            T = vector.shape[0]
            output = np.zeros((T, self.number_of_lists + 1))
            for t in range(T):
                for i in range(self.number_of_lists + 1):
                    output[t, i] = self.hit_rate(vector[t], i)
            return output

    def defineDrift(self, evaluate_at=None):
        def drift(x):
            hitRates = [self.hit_rate(x, i) for i in range(self.number_of_lists + 1)]
            dX = np.zeros(self.model_dimension)
            for i in range(self.number_of_items):
                for k in range(self.number_of_lists):
                    dX[self.index(i, k)] += (-self.p[i] * x[self.index(i, k)]
                                             + hitRates[k] * x[self.index(i, k + 1)] / self.m[k])
                    dX[self.index(i, k + 1)] += (self.p[i] * x[self.index(i, k)]
                                                 - hitRates[k] * x[self.index(i, k + 1)] / self.m[k])
            return dX

        if evaluate_at is None:
            return drift
        else:
            return drift(evaluate_at)

    def defineDriftDerivativeQ(self, evaluate_at=None):
        def Fp(x):
            hitRates = [self.hit_rate(x, i) for i in range(self.number_of_lists + 1)]
            Fp = np.zeros((self.model_dimension, self.model_dimension))
            for i in range(self.number_of_items):
                for k in range(self.number_of_lists):
                    Fp[self.index(i, k), self.index(i, k)] -= self.p[i]
                    Fp[self.index(i, k + 1), self.index(i, k)] += self.p[i]
                    Fp[self.index(i, k), self.index(i, k + 1)] += hitRates[k] / self.m[k]
                    Fp[self.index(i, k + 1), self.index(i, k + 1)] -= hitRates[k] / self.m[k]
                    for j in range(self.number_of_items):
                        Fp[self.index(i, k), self.index(j, k + 1)] -= self.p[i] * x[self.index(i, k)] / self.m[k]
                        Fp[self.index(i, k + 1), self.index(j, k + 1)] += self.p[i] * x[self.index(i, k)] / self.m[k]
                        Fp[self.index(i, k), self.index(j, k)] += self.p[j] * x[self.index(i, k + 1)] / self.m[k]
                        Fp[self.index(i, k + 1), self.index(j, k)] -= self.p[j] * x[self.index(i, k + 1)] / self.m[k]
            return Fp

        def Fpp(x):
            Fpp = np.zeros((self.model_dimension, self.model_dimension, self.model_dimension))
            for i in range(self.number_of_items):
                for k in range(self.number_of_lists):
                    for j in range(self.number_of_items):
                        if j != i:
                            Fpp[self.index(i, k), self.index(j, k), self.index(i, k + 1)] += self.p[j] / self.m[k]
                            Fpp[self.index(i, k), self.index(i, k + 1), self.index(j, k)] += self.p[j] / self.m[k]
                            Fpp[self.index(i, k), self.index(j, k + 1), self.index(i, k)] += -self.p[i] / self.m[k]
                            Fpp[self.index(i, k), self.index(i, k), self.index(j, k + 1)] += -self.p[i] / self.m[k]
                            Fpp[self.index(i, k + 1), self.index(j, k), self.index(i, k + 1)] -= self.p[j] / self.m[k]
                            Fpp[self.index(i, k + 1), self.index(i, k + 1), self.index(j, k)] -= self.p[j] / self.m[k]
                            Fpp[self.index(i, k + 1), self.index(j, k + 1), self.index(i, k)] -= -self.p[i] / self.m[k]
                            Fpp[self.index(i, k + 1), self.index(i, k), self.index(j, k + 1)] -= -self.p[i] / self.m[k]
            return Fpp

        def Q(x):
            Q = np.zeros((self.model_dimension, self.model_dimension))
            for i in range(self.number_of_items):
                for k in range(self.number_of_lists):
                    for j in range(self.number_of_items):
                        rate = self.p[i] * x[self.index(i, k)] * x[self.index(j, k + 1)] / self.m[k]
                        indices = [self.index(i, k), self.index(j, k), self.index(i, k + 1), self.index(j, k + 1)]
                        signs = [-1, 1, 1, -1]
                        for (ia, a) in enumerate(indices):
                            for (ib, b) in enumerate(indices):
                                Q[a, b] += rate * signs[ia] * signs[ib]
            return Q

        if evaluate_at is None:
            return Fp, Fpp, Q
        else:
            return Fp(evaluate_at), Fpp(evaluate_at), Q(evaluate_at)

    def convertTo2D(self, vector):
        if len(vector.shape) == 1:
            output = np.zeros((self.number_of_items, self.number_of_lists + 1))
            for i in range(self.number_of_items):
                for k in range(self.number_of_lists + 1):
                    output[i, k] = vector[self.index(i, k)]
        else:
            T = vector.shape[0]
            output = np.zeros((T, self.number_of_items, self.number_of_lists + 1))
            for t in range(T):
                for i in range(self.number_of_items):
                    for k in range(self.number_of_lists + 1):
                        output[t, i, k] = vector[t, self.index(i, k)]
        return output

    def hitRate_items(self, vector):
        if len(vector.shape) == 1:
            output = np.zeros((self.number_of_items, self.number_of_lists + 1))
            for i in range(self.number_of_items):
                for k in range(self.number_of_lists + 1):
                    output[i, k] = vector[self.index(i, k)]
        else:
            T = vector.shape[0]
            output = np.zeros((T, self.number_of_items, self.number_of_lists + 1))
            for t in range(T):
                for i in range(self.number_of_items):
                    for k in range(self.number_of_lists + 1):
                        output[t, i, k] = vector[t, self.index(i, k)]
        return output

    def meanFieldExpansionSteadyState2D(self, order=1):
        X, V, W = self.meanFieldExpansionSteadyState(order)
        return self.convertTo2D(X), self.convertTo2D(V)

        # pi  = np.zeros( (self.number_of_items,self.number_of_lists+1) )
        # V2d = np.zeros( (self.number_of_items,self.number_of_lists+1) )
        # print(pi.shape)
        # for i in range(self.number_of_items):
        #     for k in range(self.number_of_lists+1):
        #         pi[i,k]  = X[self.index(i,k)]
        #         V2d[i,k] = V[self.index(i,k)]
        # return pi,V2d

    def hit_rates2d(self, vector):
        if len(vector.shape) == 2:
            return np.array([np.sum([self.p[i] * vector[i, list_id] for i in range(self.number_of_items)])
                             for list_id in range(self.number_of_lists + 1)])
        else:
            T = vector.shape[0]
            output = np.zeros((T, self.number_of_lists + 1))
            for t in range(T):
                output[t, :] = self.hit_rates2d(vector[t])
            return output

    def simulate(self,N,time, seed_nr=None):
        """
        Simulates an realization of the stochastic process with N objects 

        Returns:
            (T,X), where : 
            - T is a 1-dimensional numpy array, where T[i] is the time of the i-th time step. 
            - x is a 2-dimensional numpy array, where x[i,j] is the j-th coordinate of the system at time T[i]
        """
        if N == 'inf':
            return self.ode(time)
        if self._x0 is None :
            raise InitialConditionNotDefined 
        nb_trans=len(self._list_of_transitions)
        t=0
    
        #if fix!=-1:     seed(fix)
        x = np.array(self._x0)
        T = [0]
        X = [x]
        if seed_nr is not None:
            rnd.seed(seed_nr)
        while t<time:
            L_rates = [self._list_of_rate_functions[i](x) for i in range(nb_trans)]
            if any(rate<-1e-14 for rate in L_rates):
               raise NegativeRate
            S=sum(L_rates)
            #print(S)
            if S<=1e-14:
                print('System stalled (total rate = 0)')
                t = time
            else:
                a=rnd.random()*S
                l=0
                while a > L_rates[l]:
                    a -= L_rates[l]
                    l += 1
        
                x = x+(1./N)*self._list_of_transitions[l]

                t+=rnd.expovariate(N*S)
            
            T.append(t)
            X.append(x)
    
        X = np.array(X)
        return(T,X)

    def meanFieldExpansionTransient(self,order=1,time=10):
        """ Computes the transient values of the mean field approximation or its O(1/N^{order})-expansions
        
        Args:
           - order : can be 0 (mean field approx.), 1 (O(1/N)-expansion) or 2 (O(1/N^2)-expansion)
        
        Returns : (T,XVW) or (T,XVWABCD), where T is a time interval and XVW is a (2d+d^2)*number_of_steps matrix (or XVWABCD is a (3n+2n^2+n^3+n^4) x number_of_steps matrix), where : 
        * XVW[0:n,:]                 is the solution of the ODE (= mean field approximation)
        * XVW[n:2*n,:]               is V(t) (= 1st order correction)
        * XVW[2*n:2*n+n**2,:]        is W(t)
        * XVWABCD[2*n+n**2,3*n+n**2] is A(t) (= the 2nd order correction)
        """
        n=len(self._list_of_transitions[0])
        t_start = ti.time()
        
        # We first defines the function that will be used to compute the drift (using symbolic computation)
        computeF = self.defineDrift()
        if (order >= 1): # We need 2 derivatives and Q to get the O(1/N)-term
            computeFp,computeFpp,computeQ = self.defineDriftDerivativeQ()
        if (order >= 2): # We need the next 2 derivatives of F and Q + the tensor R 
            computeFppp,computeFpppp,computeQp,computeQpp,computeR = self.defineDriftSecondDerivativeQderivativesR()
        print('time to compute drift=',ti.time()-t_start)
        
        if order==0:
            X_0 = self._x0
            Tmax=time
            T = np.linspace(0,Tmax,1000)
            numericalInteg = scipy.integrate.solve_ivp( lambda t,x : computeF(x), [0,Tmax], X_0,t_eval=T,rtol=1e-6)
            return(numericalInteg.t,numericalInteg.y.transpose())
        if order==1:
            XVW_0 = np.zeros(2*n+n**2)
            XVW_0[0:n] = self._x0

            Tmax=time
            T = np.linspace(0,Tmax,1000)

            numericalInteg = scipy.integrate.solve_ivp( lambda t,x : 
                                                 drift_r_vector(x,n,computeF,computeFp,computeFpp,computeQ), 
                                                  [0,Tmax], XVW_0,t_eval=T,rtol=1e-6)
            XVW = numericalInteg.y.transpose()
            X = XVW[:,0:n]
            V = XVW[:,n:2*n]
            return(numericalInteg.t,X,V,XVW)
        elif order==2:
            XVWABCD_0 = np.zeros(3*n+2*n**2+n**3+n**4)
            XVWABCD_0[0:n] = self._x0
            
            Tmax=time
            T = np.linspace(0,Tmax,1000)

            numericalInteg = scipy.integrate.solve_ivp( lambda t,x : 
                                                 drift_rr_vector(x,n,computeF,computeFp,computeFpp,computeQ,
                                                        computeFppp,computeFpppp,computeQp,computeQpp,computeR), 
                                                 [0,Tmax], XVWABCD_0,t_eval=T,rtol=1e-6)
            XVWABCD = numericalInteg.y.transpose()
            X = XVWABCD[:,0:n]
            V = XVWABCD[:,n:2*n]
            A = XVWABCD[:,2*n+n**2:3*n+n**2]
            return(numericalInteg.t,X,V,A,XVWABCD)
        else:
            print("order must be 0 (mean field), 1 (refined of order O(1/N)) or 2 (refined order 1/N^2)")


def initialize_model(nr_items, zipf_alpha, cache_sizes, initial_state):
    # set / load rate parameters
    zipf_distribution = zipf(n=nr_items, alpha=zipf_alpha)

    # initialize model with parameters and inital state
    model = cacheRANDmDDPP(zipf_distribution, cache_sizes)
    model.set_initial_state(initial_state)
    return model

class timer():
    import time as ti

    def __init__(self):
        self.t = ti.time()

    def print_timer(self, str):
        print('Time to compute', str, ti.time() - self.t, 'seconds')
        self.t = ti.time()

if __name__ == "__main__":
    n = 50
    m = [10, 5, 5]
    alpha = 0.8
    p = zipf(n, alpha)

    cacheExampleClass = cacheRANDmDDPP(p, m)

    T, X = cacheExampleClass.simulate(N=1, time=200.0)
    hit_popularities = [[] for i in range(len(m)+1)]
    for cache in range(len(m) + 1):
        print("\n-----\n")
        for i, state in enumerate(X):
            value = cacheExampleClass.hit_rate(state, cache)
            # print(value)
            hit_popularities[cache].append(value)

    import matplotlib.pyplot as plt
    plt.plot(T, hit_popularities[0], linestyle=":", color="r")
    plt.plot(T, hit_popularities[1], linestyle=":", color="g")
    plt.plot(T, hit_popularities[2], linestyle=":", color="b")
    plt.plot(T, hit_popularities[3], linestyle=":", color="y")
    plt.show()

    pass