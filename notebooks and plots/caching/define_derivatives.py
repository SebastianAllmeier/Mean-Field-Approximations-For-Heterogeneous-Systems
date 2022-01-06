import numpy as np
try:
    from numba import jit
except ImportError:
    print("Numba is not installed. We will run without it")
    print("This will be *much* slower for model of size > 100")
    def jit(nopython=True):
        def useless_decorator(func):
            return func
        return useless_decorator
    
def successor(rates):
    return np.sum(rates,0) > 0

def drift(rates, m, x):
    n = len(rates)
    h = len(m)
    return drift_jit(np.zeros((n, h)), successor(rates), rates, m, x)

@jit(nopython=True)
def drift_jit(f, is_successor, rates, m, x):
    """
    computes the drift of the system
    """
    n = len(rates)
    h = len(m)
    x_2D = x.reshape((n, h))
    for l in range(h):
        for lp in range(h):
            if is_successor[l,lp]:
                for i in range(n):
                    # i leaves list "l" for list "lp"
                    f[i,l] += - rates[i,l,lp]*x_2D[i,l]
                    f[i,lp] +=  rates[i,l,lp]*x_2D[i,l]
                    for j in range(n):
                        # j leaves list "lp" for list "l"
                        f[i,l]  += x_2D[i,lp]/m[lp] * rates[j,l,lp]*x_2D[j,l]
                        f[i,lp] -= x_2D[i,lp]/m[lp] * rates[j,l,lp]*x_2D[j,l]
    return f.reshape((n*h))

def compute_fp(rates, m, x):
    n = len(rates)
    h = len(m)
    return compute_fp_jit(np.zeros((n, h, n, h)), successor(rates), rates, m, x)

@jit(nopython=True)
def compute_fp_jit(fp, is_successor, rates, m, x):
    """
    Compute the first derivative of the drift (x is a 1D vector)
    """
    #successors = compute_successors(rates)
    n = len(rates)
    h = len(m)
    x_2D = x.reshape((n, h))
    for l in range(h):
        # We exchange i in 'l' with k in 'lp'$
        for lp in range(h):
            if is_successor[l,lp]:
                for i in range(n):
                    for k in range(n):
                        # derivative wrt x_{i,:}
                        rateDerivativeI = rates[i, l, lp]*x_2D[k, lp]/m[lp]
                        # derivative wrt x_{k,:}
                        rateDerivativeK = rates[i, l, lp]*x_2D[i, l]/m[lp]
                        for (sign, idx0, idx1) in [(-1, i, l), (1, i, lp), (1, k, l), (-1, k, lp)]:
                            fp[idx0, idx1, i, l] += sign*rateDerivativeI
                            fp[idx0, idx1, k, lp] += sign*rateDerivativeK
    return fp.reshape((n*h, n*h))

def compute_fpp(rates, m, x):
    n = len(rates)
    h = len(m)
    return compute_fpp_jit(np.zeros((n,h,n,h,n,h)), successor(rates), rates, m, x) # Second derivative

@jit(nopython=True)
def compute_fpp_jit(fpp, is_successor, rates, m, x):
    """
    Compute the second derivative of the drift at x (x is a 1D vector)
    """
    n = len(rates)
    h = len(m)
    for l in range(h):
        for lp in range(h):
            if is_successor[l,lp]:
                for i in range(n):
                    for k in range(n):
                        for (sign, idx0,idx1) in [(-1,i,l),(1,i,lp),(1,k,l),(-1,k,lp)]:
                            fpp[idx0, idx1,i,l,k,lp] += sign*rates[i,l,lp]/m[lp]
                            fpp[idx0, idx1,k,lp,i,l] += sign*rates[i,l,lp]/m[lp]
    return fpp.reshape((n*h,n*h,n*h))

def compute_q(rates, m, x):
    n = rates.shape[0]
    h = len(m)
    return compute_q_jit(np.zeros( (n,h,n,h) ), successor(rates), rates, m, x) # Co-variance Matrix Q

@jit(nopython=True)
def compute_q_jit(q, is_successor, rates, m, x):
    n = rates.shape[0]
    h = len(m)
    x_2D = x.reshape((n, h))
    for l in range(h):
        for lp in range(h):
            if is_successor[l,lp]:
                for i in range(n):
                    for k in range(n):
                        if i !=k:
                            rateTransition = rates[i, l,lp] * x_2D[i,l] * x_2D[k,lp]/m[lp]
                            for (sign0, idx00,idx01) in [(-1,i,l),(1,i,lp),(1,k,l),(-1,k,lp)]:
                                for (sign1, idx10,idx11) in [(-1,i,l),(1,i,lp),(1,k,l),(-1,k,lp)]:
                                    q[idx00, idx01,idx10,idx11] += sign0*sign1*rateTransition
    return q.reshape((n*h, n*h))

@jit(nopython=False)
def tensordot_by_fpp(W, is_successor, rates, m, h):
    n = rates.shape[0]
    res = np.zeros(n*h)
    def index(i,l):
        return i*h + l
    for l in range(h):
        for lp in range(h):
            if is_successor[l,lp]:
                for i in range(n):
                    for k in range(n):
                        for (sign, idx0,idx1) in [(-1,i,l),(1,i,lp),(1,k,l),(-1,k,lp)]:
                            res[index(idx0, idx1)] += (W[index(i,l),index(k,lp)]+W[index(k,lp),index(i,l)])* sign*rates[i,l,lp]/m[lp]
    return res
