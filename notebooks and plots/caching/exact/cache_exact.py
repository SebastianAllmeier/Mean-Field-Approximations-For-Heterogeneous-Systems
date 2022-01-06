import numpy as np 
import time

CACHED_RESULTS = dict([])
def compute_E(p, r, k):
    """
    This computes the expression for E(r, k) defined in Equation (3) of the paper

    To avoid recomputing things, we cached the results. 

    Inputs:
    - p = probability vector
    - r = sizes of lists of the cache
    - k = number of item
    """
    if tuple(p) not in CACHED_RESULTS:
        CACHED_RESULTS[tuple(p)] = dict([])
    cached_results = CACHED_RESULTS[tuple(p)]
    def E(r, k):
        """
        This subfunction applies directely the recurrence equation (4).
        """
        if sum(r) == 0:
            return 1
        if sum(r) > k or min(r) < 0:
            return 0
        key = (tuple(r), k)
        if key not in cached_results:
            h = len(r)
            def rj(j):
                rj = np.copy(r)
                rj[j] -= 1
                return rj
            cached_results[key] = E(r, k-1) + sum([r[j]*p[k-1]**(j+1)*E(rj(j), k-1) for j in range(h)])
        return cached_results[key]
    return E(r, k)

def proba_last_item(p, m):
    """
    Returns the "probability vector" of the last item
    """
    n = len(p)
    h = len(m)
    probability = np.zeros(h)
    for i in range(h):
        m1 = np.copy(m)
        m1[i] -= 1
        probability[i] = m[i]*p[-1]**(i+1)*compute_E(p, m1, n-1) / compute_E(p, m, n)
    return probability

def proba_all_item(p, m):
    """
    Returns a vector p_exact[:, :], where p_exact[i, j] is the 'exact'
    probability that item i is in list j (in steady-state). 

    The computation uses the product form solution. 
    """
    if len(p)>50:
        print("Warning: this code has not been tested for large n.")
        print("It might be slow or prone to numerical error")
    p_exact = np.zeros(shape=(len(p), len(m)))
    n = len(p)
    for i in range(n):
        new_p = np.copy(p)
        new_p[i] = p[-1]
        new_p[-1] = p[i]
        p_exact[i,:] = proba_last_item(new_p, m)
    return p_exact

def swap(v, i, limit_left, limit_right):
    tmp = np.copy(v[i:limit_left])
    v[i:limit_right] = v[limit_left:]
    v[limit_right:] = tmp

def proba_all_item_log(p, m):
    """
    Does the same as "proba_all_item". In theory this should be more efficient. 
    In practice it seems that it needs more optimization. 

    Recommendation: do not use it for now. 
    """
    n = len(p)
    newp = np.copy(p)
    indices = np.arange(n)
    p_exact = np.zeros(shape=(len(p), len(m)))
    def recursive_call(i):
        if i >= n-1:
            last_item = indices[-1]
            if p_exact[last_item, 0] == 0:
                p_exact[last_item, :] = proba_last_item(newp, m)
        else:
            limit_right = int(np.ceil((i+n)/2))
            limit_left = int(np.floor((i+n)/2))
            recursive_call(limit_left)
            swap(newp, i, limit_left, limit_right)
            swap(indices, i, limit_left, limit_right)
            recursive_call(limit_left)
            swap(newp, i, limit_right, limit_left)
            swap(indices, i, limit_right, limit_left)
    recursive_call(0)
    return p_exact

def proba_all_item_zipf(n, m, alpha):
    """
    Returns a vector p_exact[:, :], where p_exact[i, j] is the 'exact'
    probability that item i is in list j (in steady-state). 

    Parameters: 
    - n = number of items
    - m = sizes of the lists
    - alpha = parameter of the Zipf distribution. 
    """
    return proba_all_item(zipf(n, alpha), m)

def zipf(n, beta):
    p = (1+np.arange(n))**(-beta)
    return p / sum(p)

def overall_miss(p, m):
    n = len(p)
    m1 = np.copy(m)
    m1[0] += 1
    return compute_E(p, m1, n) / compute_E(p, m, n)
