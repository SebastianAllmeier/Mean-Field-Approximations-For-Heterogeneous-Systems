import numpy as np
import random as rnd
import scipy.integrate as integrate
import sympy as sym
import scipy.linalg 
import numpy.linalg
import matplotlib.pyplot as plt

from numpy import tensordot as tsdot
from sympy.utilities.lambdify import lambdify
from sympy import derive_by_array
    

import os, sys
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

# To compute the transient trajectories and fixed points 
from .refinedRefined_transientRegime import drift_r_vector, drift_rr_vector
from .refinedRefined_fixedPoint import computePiV,computePiVA

import time as ti

def eval_array_at(F,subs):
    """ evaluate an array of 'symbolic' expression using the dictionary 'subs'. 
    
    Returns : an Array of float
    """
    myF = F.flatten()
    myF = np.array([float(f.evalf(subs=subs)) for f in myF])
    return(myF.reshape(F.shape))

    
class RmfError(Exception):
    """Basic error class for this module
    """
class DimensionError(RmfError):
    pass
class NotImplemented(RmfError):
    pass
class NegativeRate(RmfError):
    pass
class InitialConditionNotDefined(RmfError):
    pass

class DDPP():
    """ DDPP serves to define and study density depend population processes
    
    """

    def __init__(self):
        """
        Explain params 
        
        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
        """
        self._list_of_transitions = []
        self._list_of_rate_functions = []
        self._x0 = None
        self._model_dimension = None

    def add_transition(self,l,f):
        """
        Add a new transition of the form (\ell,\beta_\ell), where \beta_\ell(x) is the rate of the transition

        Args:
            l: A vector of changes (of size n)
            f: A function from R^n \to R, where f(x) is the rate at which the transition occurs

        Returns:
            True if successful, False otherwise 

        Raises: 
            DimensionError if l is not of the right size.
        """
        if self._model_dimension is not None and self._model_dimension != len(l):
            raise DimensionError
        self._model_dimension = len(l)
        self._list_of_transitions.append(np.array(l))
        self._list_of_rate_functions.append(f)

    def set_initial_state(self,x0):
        if self._model_dimension and self._model_dimension != len(x0):
            raise DimensionError
        self._model_dimension = len(x0)
        self._x0 = x0

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
    
    def ode(self,time,number_of_steps=1000):
        """Simulates the ODE (mean-field approximation)
        """
        if self._x0 is None:
            raise InitialConditionNotDefined
        def drift(x):
            return (sum([self._list_of_transitions[i]*self._list_of_rate_functions[i](x) for i in range(len(self._list_of_transitions))],0))
        
        T = np.linspace(0,time,number_of_steps)
        X = integrate.odeint( lambda x,t : drift(x), self._x0, T)
        
        return(T,X)

        
    def fixed_point(self):
        """Computes the fixed of the ODE (if this ODE has a fixed point starting from x0)
        """
        if self._x0 is None:
            print('No initial condition given. We assume that the initial condition is "x0=[1,0,...]"')
            self._x0 = np.zeros(self._model_dimension)
            self._x0[0] = 1
        return(self.ode(time=10000)[1][-1,:])

    def doTransitionsConserveSum(self):
        """This function tests if the transitions conserve the sum of the coordinates.

        Returns : True or False.
        """
        for l in self._list_of_transitions:
            if sum(l) != 0:
                return False
        return(True)
    
    def dimensionReduction(self,A):
        """When the 
        
        """
        n = len(A)
        M = np.array([l for l in self._list_of_transitions])
        rank_of_transisions=np.linalg.matrix_rank(M)
        # If rank_of_transisions < n, this means that the stochastic process
        # evolves on a linear subspace of R^n. 
        eigenvaluesOfJacobian = scipy.linalg.eig(A,left=False,right=False)
        rank_of_jacobian = np.linalg.matrix_rank(A)
        if sum(numpy.real(eigenvaluesOfJacobian) < 1e-8) < rank_of_transisions:
            # This means that there are less than "rank_of_transisions" eigenvalues with <0 real part
            print("The Jacobian seems to be not Hurwitz")
        if  rank_of_jacobian == n:
            return(np.eye(n),np.eye(n),n)
        C = np.zeros((n,n))
        n = len(A)
        d = 0
        rank_of_previous_submatrix = 0
        for i in range(n):
            rank_of_next_submatrix = np.linalg.matrix_rank(A[0:i+1,0:i+1])
            if  rank_of_next_submatrix > rank_of_previous_submatrix:
                C[d,i] = 1
                d+=1
            rank_of_previous_submatrix = rank_of_next_submatrix
        U,s,V = scipy.linalg.svd(A)
        C[rank_of_jacobian:,:] = U.transpose()[rank_of_jacobian:,:]
        return(C,scipy.linalg.inv(C),rank_of_jacobian)

    def test_for_linear_dependencies(self):
        """This function tests if there the transition conserve some subset of the space. 
        
        Return (dim, subs) where
        * dim is the rank of the matrix of transitions
        * subs corresponds to the null space of the transitions"""
        
        M = np.array([l for l in self._list_of_transitions])
        u,d,v = np.linalg.svd(M,full_matrices=False)
        dim = sum(abs(d) > 1e-8)
        if dim < self._model_dimension:
            v = v[dim:]
            p,l,u = scipy.linalg.lu(v)
            subs=u
            for i in range(subs.shape[0]):
                j = np.argmax( abs(subs[i,:])>1e-5)
                subs[i,:] = subs[i,:] / subs[i,j]
        else:
            subs = np.array([])
        return(dim,subs)

    def _numericalJacobian(self,drift,fixedPoint,epsilon=1e-6):
        """ Computes the Jacobian of the drift using a finite difference method : 
        
        Returns : dF[i][j] = \partial f_i / \partial x_j evaluates in fixedPoint
        """
        dim = len(fixedPoint)
        def e(i):
            res = np.zeros(dim)
            res[i] = 1
            return(res)
        A = [(drift(fixedPoint+epsilon*e(j))-drift(fixedPoint-epsilon*e(j)))/(2*epsilon) for j in range(dim)]
        return(np.array(A).transpose())

    
    def _numericalHessian(self,drift,fixedPoint,epsilon=1e-4):
        """ Computes the Jacobian of the drift using a finite difference method : 
        
        Returns : ddF[i][j][k] = \partial^2 f_i / (\partial x_j\partial x_k) evaluates in fixedPoint
        """
        dim = len(fixedPoint)
        def e(i):
            res = np.zeros(dim)
            res[i] = 1
            return(res)
        def ee(i,j):
            res = np.zeros(dim)
            res[i] = 1
            res[j] += 1
            return(res)
        ffixedPoint = drift(fixedPoint)
        ddB = [[drift(fixedPoint+epsilon*ee(i,j)) for j in range(i+1)] for i in range(dim)]
        dB = [drift(fixedPoint+epsilon*e(i)) for i in range(dim)]
        B = [[(ddB[max(i,j)][min(i,j)] - dB[i] - dB[j] + ffixedPoint)/epsilon**2
              for i in range(dim)] for j in range(dim)]
        B = [[[B[j][k][l] for j in range(dim)] for k in range(dim)] for l in range(dim)]
        return(np.array(B))


    def defineDrift(self,evaluate_at=None):
        """Return a (lambdified) function F(.) that is the drift of the system

        Arg : evaluate_at=None : if not None, return the value of the drift at point "evaluate_at".
        """
        n=len(self._list_of_transitions[0])
        number_of_transitions = len(self._list_of_transitions)
        x = [sym.symbols('x[{}]'.format(i)) for i in range(n)]
        f=np.zeros(n)
        for l in range(number_of_transitions):
            f = f + self._list_of_transitions[l]*self._list_of_rate_functions[l](x)
        F = sym.lambdify([x],[f[i] for i in range(n)])
        def computeF(x):     return(np.array(F(x)))
        return computeF

    def defineDriftDerivativeQ(self,evaluate_at=None):
        """Return three (lambdified) functions Fp(.), Fpp(.) and Q(.) that are the first two derivatives of the drift and the matrix Q. 

        Arg : evaluate_at=None : if not None, return the value of the functions evaluated at point "evaluate_at".
        """
        n=len(self._list_of_transitions[0])
        number_of_transitions = len(self._list_of_transitions)
        x = [sym.symbols('x[{}]'.format(i)) for i in range(n)]
        f=np.zeros(n)
        for l in range(number_of_transitions):
            f = f + self._list_of_transitions[l]*self._list_of_rate_functions[l](x)
        dF = np.array([[sym.diff(f[i],x[j]) for j in range(n)] for i in range(n)])
        ddF = np.array([[[sym.diff(dF[i,j],x[k]) for j in range(n)]for k in range(n)] for i in range(n)])

        if evaluate_at is not None:
            subs_dictionary = {x[i]:evaluate_at[i] for i in range(n)}
            Fp = eval_array_at(dF,subs_dictionary)
            Fpp = eval_array_at(ddF,subs_dictionary)
            Q = np.zeros((n**2))
            for l in range(number_of_transitions):
                Q = Q + np.kron(self._list_of_transitions[l],self._list_of_transitions[l])*self._list_of_rate_functions[l](evaluate_at)
            return(Fp,Fpp,Q.reshape((n,n)))

        q = np.zeros((n**2))
        for l in range(number_of_transitions):
            q = q + np.kron(self._list_of_transitions[l],self._list_of_transitions[l])*self._list_of_rate_functions[l](x)
            
        Fp = sym.lambdify([x],list(dF.reshape(n**2)))
        Fpp = sym.lambdify([x],list(ddF.reshape((n**3))))
        Q = sym.lambdify([x],[q[i] for i in range(n*n)])
        
        def computeFp(x):    return(np.array(Fp(x)).reshape( (n,n) ))
        def computeFpp(x):   return(np.array(Fpp(x)).reshape( (n,n,n) ))
        def computeQ(x):     return(np.array(Q(x)).reshape( (n,n) ))
        return( computeFp,computeFpp,computeQ )
    

    
    def defineDriftSecondDerivativeQderivativesR(self,evaluate_at=None):
        """Return three (lambdified) functions Fppp(.), Fpppp(.) and Q(.) that are the 3rd and 4th derivatives of the drift, the first two derivatives of Q(.) and the matrix R(.)
        
        Arg : evaluate_at=None : if not None, return the value of the functions evaluated at point "evaluate_at".
        """
        n=len(self._list_of_transitions[0])
        number_of_transitions = len(self._list_of_transitions)
        x = [sym.symbols('x[{}]'.format(i)) for i in range(n)]
        f=np.zeros(n)
        for l in range(number_of_transitions):
            f = f + self._list_of_transitions[l]*self._list_of_rate_functions[l](x)
        dddF = np.array([[sym.diff(f[i],x[j],x[k],x[l]) for j in range(n) for k in range(n) for l in range(n)] 
                          for i in range(n)]).reshape((n**4))
        dddF_nonReshaped=dddF.reshape((n,n,n,n))
        ddddF = np.array([[sym.diff(dddF_nonReshaped[i,j,k,l],x[m]) for j in range(n) for k in range(n) for l in range(n) for m in range(n)] 
                           for i in range(n)]).reshape((n**5))
        r = np.zeros((n**3))
        for l in range(number_of_transitions):
            r = r + np.kron(self._list_of_transitions[l],np.kron(self._list_of_transitions[l],self._list_of_transitions[l]))*self._list_of_rate_functions[l](x)
        q = np.zeros((n**2))
        for l in range(number_of_transitions):
            q = q + np.kron(self._list_of_transitions[l],self._list_of_transitions[l])*self._list_of_rate_functions[l](x)
        dQ = np.array( [[[sym.diff(q[i],x[k]) for k in range(n)] for i in range(n**2)]] ).reshape(n**3)
        ddQ = np.array( [[[sym.diff(q[i],x[k],x[l]) for k in range(n) for l in range(n)] for i in range(n**2)]] ).reshape(n**4)

        if evaluate_at is not None:
            subs_dictionary = {x[i]:evaluate_at[i] for i in range(n)}
            Fppp = eval_array_at(dddF.reshape((n,n,n,n)),subs_dictionary)
            Fpppp = eval_array_at(ddddF.reshape((n,n,n,n,n)),subs_dictionary)
            Qp = eval_array_at(dQ.reshape((n,n,n)),subs_dictionary)
            Qpp = eval_array_at(ddQ.reshape((n,n,n,n)),subs_dictionary)
            R = eval_array_at(r.reshape((n,n,n)),subs_dictionary)
            return(Fppp,Fpppp,Qp,Qpp,R)
        
        Fppp = sym.lambdify([x],[dddF[i] for i in range(n**4)])
        Fpppp = sym.lambdify([x],[ddddF[i] for i in range(n**5)])
        Qp = sym.lambdify([x],[dQ[i] for i in range(n**3)])
        Qpp = sym.lambdify([x],[ddQ[i] for i in range(n**4)])
        R = sym.lambdify([x],[r[i] for i in range(n**3)])
        def computeFppp(x):  return(np.array(Fppp(x)).reshape( (n,n,n,n) ))
        def computeFpppp(x): return(np.array(Fpppp(x)).reshape( (n,n,n,n,n) ))
        def computeQp(x):    return(np.array(Qp(x)).reshape( (n,n,n) ))
        def computeQpp(x):   return(np.array(Qpp(x)).reshape( (n,n,n,n) ))
        def computeR(x):     return(np.array(R(x)).reshape( (n,n,n) ))
        return(computeFppp,computeFpppp,computeQp,computeQpp,computeR)
    
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
            numericalInteg = integrate.solve_ivp( lambda t,x : computeF(x), [0,Tmax], X_0,t_eval=T,rtol=1e-6)
            return(numericalInteg.t,numericalInteg.y.transpose())
        if order==1:
            XVW_0 = np.zeros(2*n+n**2)
            XVW_0[0:n] = self._x0

            Tmax=time
            T = np.linspace(0,Tmax,1000)

            numericalInteg = integrate.solve_ivp( lambda t,x : 
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

            numericalInteg = integrate.solve_ivp( lambda t,x : 
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

    def reduceDimensionFpFppQ(self,Fp,Fpp,Q):
        P,Pinv, rank=self.dimensionReduction(Fp)
        Fp = (P@Fp@Pinv)[0:rank,0:rank]
        Fpp = tsdot(tsdot(tsdot(P,Fpp,1), Pinv,1),Pinv,axes=[[1],[0]])[0:rank,0:rank,0:rank]
        Q = (P@Q@P.transpose())[0:rank,0:rank]
        return(Fp,Fpp,Q, P, Pinv,rank)

    def expandDimensionVW(self,V,W,Pinv):
        rank = len(V)
        return(Pinv[:,0:rank]@ V, Pinv[:,0:rank]@W@Pinv.transpose()[0:rank,:])

    def reduceDimensionFppFppppQpQppR(self, Fppp,Fpppp,Qp,Qpp,R,P,Pinv,rank):
        FpppOLD = Fppp
        FppppOLD = Fpppp
        QpOLD = Qp
        Fppp = tsdot(tsdot(tsdot(tsdot(P,Fppp,1), Pinv,1),Pinv,axes=[[1],[0]]),Pinv,axes=[[2],[0]])[0:rank,0:rank,0:rank,0:rank]
        Fpppp = tsdot(tsdot(tsdot(tsdot(tsdot(P,Fpppp,1), Pinv,1),Pinv,axes=[[1],[0]]),Pinv,axes=[[2],[0]]),Pinv,axes=[[3],[0]])[0:rank,0:rank,0:rank,0:rank,0:rank]
        Qp_new = np.transpose(tsdot(tsdot(P,np.transpose(Qp,axes=[0,2,1]),1),P.transpose(),1),axes=[0,2,1])
        Qp_final = tsdot(Qp_new,Pinv,1)[0:rank,0:rank,0:rank]
        Qpp_new = np.transpose(tsdot(tsdot(P,np.transpose(Qpp,axes=[0,3,2,1]),1),P.transpose(),1),axes=[0,3,2,1])
        Qpp = tsdot(tsdot(Qpp_new,Pinv,axes=[[2],[0]]),Pinv,axes=[[3],[0]])[0:rank,0:rank,0:rank,0:rank]
        R = tsdot(tsdot(tsdot(P,R,1),P.transpose(),axes=[[1],[0]]),P.transpose(),1)[0:rank,0:rank,0:rank]
        return(Fppp,Fpppp,Qp_final,Qpp,R)
        
    def expandDimensionABCD(self,A,B,C,D,Pinv):
        rank = len(A)
        Pinv=Pinv[:,0:rank]
        return(Pinv@A, Pinv@B@Pinv.transpose(),
               tsdot(tsdot(tsdot(C,Pinv,axes=[[2],[1]]),Pinv,axes=[[1],[1]]),Pinv,axes=[[0],[1]]),
               tsdot(tsdot(tsdot(tsdot(D,Pinv,axes=[[3],[1]]),Pinv,axes=[[2],[1]]),Pinv,axes=[[1],[1]]),Pinv,axes=[[0],[1]]))
        
    def meanFieldExpansionSteadyState(self,order=1):
        """This code computes the O(1/N) and O(1/N^2) expansion of the mean field approximaiton (the term "V" is the "V" of Theorem~1 of https://hal.inria.fr/hal-01622054/document. 
        
        Note : Probably less robust and slower that theoretical_V
        
        """
        pi = self.fixed_point()
        if order == 0:
            return pi
        if (order >= 1): # We need 2 derivatives and Q to get the O(1/N)-term
            Fp,Fpp,Q = self.defineDriftDerivativeQ(evaluate_at=pi)
            Fp,Fpp,Q, P,Pinv,rank = self.reduceDimensionFpFppQ(Fp,Fpp,Q)
            if order==1:
                pi,V,(V,W) = computePiV(pi, Fp,Fpp, Q)
                V,W = self.expandDimensionVW(V,W,Pinv)
                return(pi,V,(V,W))
        if (order >= 2): # We need the next 2 derivatives of F and Q + the tensor R
            if len(self._x0) >= 11:
                print("*Warning* The computation time grows quickly with the number of dimensions", 
                      "(probably around {:1.0f}sec. for this model)".format(0.0001*len(self._x0)**5))
            Fppp,Fpppp,Qp,Qpp,R = self.defineDriftSecondDerivativeQderivativesR(evaluate_at=pi)
            Fppp,Fpppp,Qp,Qpp,R = self.reduceDimensionFppFppppQpQppR(Fppp,Fpppp,Qp,Qpp,R,P,Pinv,rank)
            pi,V,A,(V,W,A,B,C,D) = computePiVA(pi, Fp,Fpp,Fppp,Fpppp, Q,Qp,Qpp, R)
            V,W = self.expandDimensionVW(V,W,Pinv)
            A,B,C,D = self.expandDimensionABCD(A,B,C,D,Pinv)
            if order==2: return(pi,V,A,(V,W,A,B,C,D))
            else:  print('order should be 0, 1 or 2')
    
    def theoretical_V(self, symbolic_differentiation=True):
        """This code computes the constant "V" of Theorem~1 of https://hal.inria.fr/hal-01622054/document 
        
        Note : for now this function does not support rates that depend on N (i.e. C=0)
        
        This function is deprecated (meanFieldExpansionSteadyState(order=1) should be called instead) 

        Args: 
            symbolic_differentiation (bool, default=True) : when True, the derivative are computed using Sympy. When false, they are computed using a finite difference method. 
        """
        n = self._model_dimension
        number_transitions = len(self._list_of_transitions)
        fixedPoint = self.fixed_point()
    
        Var=np.array([sym.symbols('x_{}'.format(i)) for i in range(n)])
        f_x=np.zeros(n)
        for i in range(number_transitions):
            f_x = f_x + self._list_of_transitions[i]*self._list_of_rate_functions[i](Var)
        
        # The following code attempts to reduce the number of dimensions by testing
        # if the transitions conserve some linear combinations of the coordinates
        (dim,subs) = self.test_for_linear_dependencies()
        variables = [i for i in range(n)]
        for sub in range(subs.shape[0]):
            j = np.argmax( abs(subs[sub,:])>1e-5)
            variables.remove(j)
            for i in range(n):
                f_x[i]=f_x[i].subs(Var[j],sum([self._x0[k]*subs[sub,k] for k in range(j,n)])
                                   -sum(np.array([Var[k]*subs[sub,k] for k in range(j+1,n)])))
        Var = [Var[i] for i in variables]

        if (symbolic_differentiation):
            dF = [[sym.diff(f_x[variables[i]],Var[j]) for j in range(dim)] for i in range(dim)]
            subs_dictionary = {Var[i]:fixedPoint[variables[i]] for i in range(dim)}
            A=np.array([[float(dF[i][j].evalf(subs=subs_dictionary))
                         for j in range(dim)]
                        for i in range(dim)])
            B=np.array([[[float(sym.diff(dF[j][k],Var[l]).evalf(subs=subs_dictionary) ) 
                          for l in range(dim)]
                         for k in range(dim)]
                        for j in range(dim)])
        else:
            drift = lambdify([Var], [f_x[variables[i]] for i in range(dim)])
            drift_array = lambda x : np.array(drift(x))
            fixedPointProj = np.array([fixedPoint[variables[i]] for i in range(dim)])
            A = self._numericalJacobian(drift_array,fixedPointProj)
            B = self._numericalHessian(drift_array,fixedPointProj)

        Q=np.zeros((dim,dim))
        for l in range(number_transitions):
            v = [self._list_of_transitions[l][variables[p]] for p in range(dim)]
            Q += np.kron(v,v).reshape(dim,dim)*self._list_of_rate_functions[l](fixedPoint)

        W = scipy.linalg.solve_continuous_lyapunov(A,Q)
        V = tsdot(scipy.linalg.inv(A),tsdot(B,W/2,2),1)

        # We now attemps to reconstruct the full C if the number of dimensions was reduced. 
        if dim < n :
            newV = np.zeros(n)
            for i in range(dim):
                newV[variables[i]] = V[i]
            for sub in reversed(range(len(subs))):
                j = np.argmax( abs(subs[sub,:])>1e-5)
                newV[j] = -sum(np.array([newV[k]*subs[sub,k] for k in range(j+1,n)]))
            V = newV
        return(np.array(V))


    def _batch_meanConfidenceInterval(self,T,X):
        n = len(T)
        if len(T)<300:  # We do not do batch if we have less than 200 samples
            return(np.sum(np.diff(T[int(n/2):n])*X[int(n/2):n-1]) / (T[n-1]-T[int(n/2)]),None)
        # Otherwise we split the data into 100 batchs of size 
        Y = [np.sum(np.diff(T[int(2*i*n/300):int((2*i+1)*n/300)])*
                    X[int(2*i*n/300):int((2*i+1)*n/300)-1] /
                    (T[int((2*i+1)*n/300)]-T[int(2*i*n/300)])) for i in range(50,150)]
        return (np.mean(Y),2*np.std(Y)/10)
    
                                      
    def steady_state_simulation(self,N,time=1000):
        """Generates a sample of E[X] in steady-state.

        The expectation is computed by performing one simulation from
        t=0 to t=time and averaging the values over time/2 .. time
        (the implicit assumption is that the system should be roughly
        in steady-state at time/2).

        """
        T,X = self.simulate(N,time)
        n = len(T)
        n2 = int(n/2)
        result = np.array([self._batch_meanConfidenceInterval(T,X[:,i]) for i in range(len(X[0,:]))])
        return(result[:,0],result[:,1])
        
    def compare_refinedMF(self,N,time=1000):
        """Compare E[X] with its mean-field and refined mean-field approximation  

        Args : 
             N (int) : system's size
             time : Computes the expectation by

        Return : (Xm,Xrmf,Xs,Vs) where: 
             Xm is the fixed point of the mean-field approximation 
             Xrmf = Xm + V/N
             Xs is an approximation E[X] (computed by simulation)
             Vs is an estimation of the variance
        """
        Xm = self.fixed_point()
        V = self.theoretical_V()
        Xrmf = Xm+V/N
        Xs,Vs = self.steady_state_simulation(N,time)
        return(Xm,Xrmf,Xs,Vs)
    
    def plot_ODE_vs_simulation(self,N,time=10):
        """Plot the ODE and the simulation on the same graph
        
        The system is simulated from t=0 to t=time (both for the
        stochastic and ODE version), starting from x0. 
        
        Args: 
             N (int) : system's size
             time : the time until the simulation should be run

        """
        T,X = self.simulate(N,time=10)
        plt.plot(T,X)
        plt.gca().set_prop_cycle(None)
        T,X = self.ode(time=10)
        plt.plot(T,X,'--')
        plt.legend(['x_{}'.format(i) for i in range(self._model_dimension)])
        plt.xlabel('Time')
        plt.ylabel('x_{i}')
        plt.show()
