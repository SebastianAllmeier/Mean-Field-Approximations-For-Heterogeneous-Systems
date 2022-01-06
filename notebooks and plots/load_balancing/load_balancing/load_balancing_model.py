
import scipy.integrate as integrate
import numpy as np
import random as rnd
import math
import load_balancing.jit_functions as efficient
import sys
from time import time


class lbm():
    def __init__(self, arrival_rate, server_rates, max_q=15, x0=None, verbose=False):

        # specifications
        self.n = server_rates.shape[0]
        self.dimension = self.n
        self.arrival_rate = arrival_rate
        self.mu = server_rates
        self.max_q = max_q
        self.dimensions = (self.n, self.max_q+1)
        self.verbose = verbose
        if self.verbose:
            print("Initialized model with:\n\tn={}\n\tArrival rate={}".format(self.n, arrival_rate))

        self.set_initial_state(x0=x0)

        self._dd_drift = None

        if arrival_rate >= (np.sum(self.mu) / self.n) and self.verbose:
            print("Arrival rate ({0}) is higher than average server rate ({1:0.2f}).".format(
                arrival_rate, np.sum(self.mu) / self.n))
            print("-----------------------")
        elif self.verbose:
            print("Arrival Rate ({0}); Average Server Rate ({1:0.2f}).".format(
                arrival_rate, np.sum(self.mu) / self.n))

    def set_initial_state(self, x0=None):
        if x0 is None:
            x0 = np.zeros(shape=(self.n, self.max_q + 1))
            x0[:, 0] = 1
            self.x0 = x0
            if self.verbose:
                print("Model initial state set to:\n\tAll servers have no jobs (x[:,0]=1).")
        else:
            self.x0 = x0
            print("Custom initial state set.")

    def drift(self, x):
        return efficient.efficient_drift(x, self.arrival_rate, self.mu)

    # def drift2(self, x):
    #     """
    #     Alternative drift (probably less efficient)
    #     """
    #     n = self.n
    #     _drift = np.zeros(shape=self.x0.shape)
    #
    #     for server1 in range(n):
    #         # adding a job
    #         for server2 in range(n):
    #             for s1 in range(0, self.max_q):
    #                 for s2 in range(0, self.max_q):
    #                     rate = self.arrival_rate * x[server1,s1]*x[server2,s2]/n
    #                     if s1 <= s2:
    #                         _drift[server1, s1]   -= rate
    #                         _drift[server1, s1+1] += rate
    #                     else:
    #                         _drift[server2, s2]   -= rate
    #                         _drift[server2, s2+1] += rate
    #     for server in range(n):
    #         for j in range(1, self.max_q+1):
    #             rate = x[server, j] * self.mu[server]
    #             _drift[server, j]   -= rate
    #             _drift[server, j-1] += rate
    #     return _drift

    def ode(self, time, number_of_steps=1000):
        """
        Simulates the ODE (mean-field approximation) for the model.
        """

        # Look for and load saved calculations.
        # try:
        #     ode_X = np.loadtxt('saves/{0}_{1}_{2}_{3:1.4f}_X.txt'.format(self.n, self.arrival_rate, time,
        #                                                                  np.sum(self.mu)/len(self.mu))
        #                        ).reshape((1000, self.dimensions[0], self.dimensions[1]))
        #     ode_T = np.loadtxt('saves/{0}_{1}_{2}_{3:1.4f}_T.txt'.format(self.n, self.arrival_rate, time,
        #                                                                  np.sum(self.mu)/len(self.mu)))
        #     print("ODE Results loaded from cache.")
        #     return ode_T, ode_X
        # except FileNotFoundError:
        #     pass
        # except OSError:
        #     pass

        # Check initial condition.
        if self.x0 is None:
            raise AssertionError

        # Define wrapper of the drift function for integration.
        def _drift(x):
            return efficient.efficient_drift(x, self.arrival_rate, self.mu)
        # Time steps
        T = np.linspace(0, time, number_of_steps)
        print("Integrating MF")
        # Integrate the drift.
        X = integrate.odeint(lambda x, t: _drift(x.reshape(self.dimensions)).flatten(), self.x0.flatten(), T)

        # Reshape results from array into matrix form.
        X = np.array([X[i, :].reshape(self.dimensions) for i in range(X.shape[0])])

        # Save results.
        np.savetxt('saves/{0}_{1}_{2}_{3:1.4f}_T.txt'.format(self.n,
                                                       self.arrival_rate, time, self.mu[0]), T)

        np.savetxt('saves/{0}_{1}_{2}_{3:1.4f}_X.txt'.format(self.n,
                                                       self.arrival_rate, time, self.mu[0]), X.flatten())
        return T, X

    def defineDriftDerivativeQ(self, x):
        n = x.shape[0]
        q_len = x.shape[1]

        # start_q = time()
        _q = efficient.qMatrix(x, self.arrival_rate, self.mu)
        # print("Q time: ", time() - start_q)
        # _q = self.qMatrix(x)
        # start_d = time()
        _d_drift = efficient.jacobian(x, self. arrival_rate, self.mu)
        # print("D time: ", time() - start_d)
        # _d_drift = self.jacobian(x)
        if self._dd_drift is None:
            # _dd_drift = self.hessian(x)
            _dd_drift = efficient.hessian(x, self.arrival_rate)
            self._dd_drift = _dd_drift
        else:
            # _dd_drift = efficient.hessian(x, self.arrival_rate)
            _dd_drift = self._dd_drift
        return _q, _d_drift, _dd_drift

    def drift_r_vector(self, state, dimensions, arrival_rate, mu, t, Tmax):
        # Initialize variables.
        n = dimensions[0]
        q_len = dimensions[1]
        _x = state[0:n*q_len].reshape((n, q_len))
        _v = state[n*q_len:2*n*q_len].reshape((n,q_len))
        _w = state[2*n*q_len:].reshape((n, q_len, n, q_len))

        # Initialize derivative array.
        _d_state = np.zeros(shape=state.shape, dtype=np.float64)

        # Calculate drift, first, and second derivatives and q.
        _drift = efficient.efficient_drift(_x, arrival_rate, mu)
        _q, _d_drift, _dd_drift = self.defineDriftDerivativeQ(_x)


        # Calculate the derivatives of V and W.
        _dV = np.tensordot(_d_drift, _v, axes=([2,3], [0,1])) + np.tensordot(_dd_drift, _w, axes=([2,3,4,5], [0,1,2,3])) / 2
        _dW = np.tensordot(_d_drift, _w, axes=([2,3], [0,1]))
        _dW += np.transpose(_dW, axes=[2,3,0,1])
        _dW += _q

        # Resize the tensors to a vector.
        _d_state[0:n*q_len] = _drift.flatten()
        _d_state[n*q_len:2*n*q_len] = _dV.flatten()
        _d_state[2 * n * q_len:] = _dW.flatten()

        # Print current integration time.
        if t >= 0:
            print('\r', "Integration Time: {:1.1f} / {}".format(t, Tmax), end='')
        return _d_state

    def expansionTransient(self, time):
        """

        :param time: time up to which to integrate
        :return:
        """
        # The solution for x, v, and w will be solved simultaneously.
        # x0 = initial state, v0 = 0, w0 = 0
        # Entries 0:n*q_len correspond to x.
        # Entries n*q_len:2*n*q_len correspond to v.
        # Remaining entries correspond to w.

        # Get dimensions
        n = self.dimensions[0]
        q_len = self.dimensions[1]

        # Set initial state for the ivp.
        state0 = np.zeros(2*n*q_len + (n*q_len)**2)
        state0[0:n*q_len] = self.x0.flatten()

        # Set timeframe / integration steps.
        Tmax = time
        T = np.linspace(0, Tmax, 200)

        # Set function to integrate giving dx, dv, dw.
        _drift_r_vector = lambda t, x: self.drift_r_vector(x, self.dimensions, self.arrival_rate, self.mu, t, Tmax)
        print("Integrating RMF")
        # Solve the IVP.
        numericalInteg = integrate.solve_ivp(_drift_r_vector, [0, Tmax], state0, t_eval=T, rtol=1e-6)
        print('')
        # Reshape the results
        XVW = numericalInteg.y.transpose()
        # Get solution of X.
        X = XVW[:, 0:n * q_len]
        X = np.array([X[i, :].reshape(self.dimensions) for i in range(X.shape[0])])
        # Get solution of V.
        V = XVW[:, (n * q_len) : (2 * n * q_len)]
        V = np.array([V[i, :].reshape(self.dimensions) for i in range(V.shape[0])])
        # Get times.
        T = numericalInteg.t
        return T, X, V


    def efficient_simulation(self, steps=5e5, seed_nr=-1):
        # seed=-1 -> no seeding
        x0 = np.sum(self.x0[:, 1:], axis=1)
        sim_T, sim_X = efficient.simulate_jit(x0, self.arrival_rate, self.mu, steps, seed_nr=seed_nr)
        return sim_T, sim_X

    # def time_average(self, steps=2e6, seed_nr=-1):
    #     x0 = np.sum(self.x0[:, 1:], axis=1)
    #     return calculate_time_average(x0, self.arrival_rate, self.mu, steps, seed_nr=seed_nr)

    def many_simulations(self, number_of_simulations, steps=1e4):
        """
        Estimation of transient state distribution + confidence interval.

        Output: (x, c), where:
        - x is of size N*len(M)
        - error1[i,j] is the absolute error of x[i,j]    (of one simulation vs the mean)
        - error2[i,j] is the mean square error of x[i,j] (of one simulation vs the mean)
        """
        # TODO: implement poisson probability
        mean = np.zeros((int(steps), self.dimensions[1]))
        error1 = np.zeros((int(steps), self.dimensions[1]))
        error2 = np.zeros((int(steps), self.dimensions[1]))
        print("2*{} Simulations will be run".format(number_of_simulations))
        print("Calculating Mean")
        for seed in range(number_of_simulations):
            print('\r', "Running Simu. Nr.{} of {}".format(seed+1, 2*number_of_simulations), end='')
            sim_T, sim_X = self.efficient_simulation(steps=steps, seed_nr=seed)
            sim_T, percentange_sim_X = efficient.percentage_state_representation(sim_T, sim_X, self.dimensions[1])
            mean += percentange_sim_X / number_of_simulations
        print("\nCalculating Errors (with same seeds)")
        for seed in range(number_of_simulations):
            print('\r', "Running Simu. Nr.{} of {}".format(number_of_simulations + seed+1, 2*number_of_simulations), end='')
            sim_T, sim_X = self.efficient_simulation(steps=steps, seed_nr=seed)
            sim_T, percentange_sim_X = efficient.percentage_state_representation(sim_T, sim_X, self.dimensions[1])
            error1 += np.abs(percentange_sim_X - mean) / (number_of_simulations - 1)
            error2 += np.power((percentange_sim_X - mean), 2) / (number_of_simulations - 1)
        print('')

        rates_sum = self.arrival_rate * self.n + np.sum(self.mu)
        sim_T = np.linspace(0, steps / rates_sum, int(steps))
        return sim_T, mean, error1, error2

    def many_simulations_avg_queue_len(self, number_of_simulations, steps=1e4):
        """
        Estimation of transient state distribution + confidence interval.

        Output: (x, c), where:
        - x is of size N*len(M)
        - error1[i,j] is the absolute error of x[i,j]    (of one simulation vs the mean)
        - error2[i,j] is the mean square error of x[i,j] (of one simulation vs the mean)
        """
        # TODO: implement poisson probability
        mean = np.zeros((int(steps), ))
        error1 = np.zeros((int(steps), ))
        error2 = np.zeros((int(steps), ))
        print("Many simulations for avg. queue len calculation.")
        print("2*{} Simulations will be run".format(number_of_simulations))
        print("Calculating Mean")
        for seed in range(number_of_simulations):
            print('\r', "Running Simu. Nr.{} of {}".format(seed+1, 2*number_of_simulations), end='')
            sim_T, sim_X = self.efficient_simulation(steps=steps, seed_nr=seed)
            sim_T, percentange_sim_X = efficient.percentage_state_representation(sim_T, sim_X, q_max=self.dimensions[1])
            percentage_sim_avg_X = np.sum(percentange_sim_X * np.arange(0, self.dimensions[1]), axis=1)
            mean += percentage_sim_avg_X / number_of_simulations
        print("\nCalculating Errors (with same seeds)")
        for seed in range(number_of_simulations):
            print('\r', "Running Simu. Nr.{} of {}".format(number_of_simulations + seed+1, 2*number_of_simulations), end='')
            sim_T, sim_X = self.efficient_simulation(steps=steps, seed_nr=seed)
            sim_T, percentange_sim_X = efficient.percentage_state_representation(sim_T, sim_X, q_max=self.dimensions[1])
            percentage_sim_avg_X = np.sum(percentange_sim_X * np.arange(0, self.dimensions[1]), axis=1)
            error1 += np.abs(percentage_sim_avg_X - mean) / (number_of_simulations - 1)
            error2 += np.power((percentage_sim_avg_X - mean), 2) / (number_of_simulations - 1)

        rates_sum = self.arrival_rate * self.n + np.sum(self.mu)
        sim_T = np.linspace(0, steps / rates_sum, int(steps))
        return sim_T, mean, error1, error2


    def many_simulations_single(self, number_of_simulations, steps=1e4):
        """
        Estimation of transient distribution + confidence interval.

        Output: (x, c), where:
        - x is of size N*len(M)
        - error1[i,j] is the absolute error of x[i,j]    (of one simulation vs the mean)
        - error2[i,j] is the mean square error of x[i,j] (of one simulation vs the mean)
        """
        # TODO: implement poisson probability
        mean = np.zeros((int(steps), self.dimensions[0], self.dimensions[1]))
        error1 = np.zeros((int(steps), self.dimensions[0], self.dimensions[1]))
        error2 = np.zeros((int(steps), self.dimensions[0], self.dimensions[1]))
        print("2*{} Simulations will be run".format(number_of_simulations))
        print("Calculating Mean")
        for seed in range(number_of_simulations):
            print('\r', "Running Simu. Nr.{} of {}".format(seed+1, 2*number_of_simulations), end='')
            sim_T, sim_X = self.efficient_simulation(steps=steps, seed_nr=seed)
            sim_T, percentange_sim_X = efficient.percentage_state_representation_single(sim_T, sim_X, self.dimensions[1])
            mean += percentange_sim_X / number_of_simulations
        print("\nCalculating Errors (with same seeds)")
        for seed in range(number_of_simulations):
            print('\r', "Running Simu. Nr.{} of {}".format(number_of_simulations + seed+1, 2*number_of_simulations), end='')
            sim_T, sim_X = self.efficient_simulation(steps=steps, seed_nr=seed)
            sim_T, percentange_sim_X = efficient.percentage_state_representation_single(sim_T, sim_X, self.dimensions[1])
            error1 += np.abs(percentange_sim_X - mean) / (number_of_simulations - 1)
            error2 += np.power((percentange_sim_X - mean), 2) / (number_of_simulations - 1)
        print('')

        rates_sum = self.arrival_rate * self.n + np.sum(self.mu)
        sim_T = np.linspace(0, steps / rates_sum, int(steps))
        return sim_T, mean, error1, error2


    def alternate_state_representation_2(self, state_array):
        def transform_state(state):
            # get array displaying number of servers with equal queue lengths
            max_queue_length = state.shape[1]
            new_state = np.zeros(shape=(max_queue_length))
            for i in range(max_queue_length):
                # for each queue length sum over servers
                new_state[i] = np.sum(state[:, i])
            return new_state/self.n

        if len(state_array.shape) <= 2:
            return transform_state(state_array)
        else:
            new_state_array = []
            for state in state_array:
                new_state_array.append(transform_state(state))
            new_state_array = np.array(new_state_array)
            return new_state_array

    def meanFieldExpansionSteadyState(self, order=1):
        """This code computes the O(1/N) and O(1/N^2) expansion of the mean field approximaiton
        (the term "V" is the "V" of Theorem~1 of https://hal.inria.fr/hal-01622054/document.

        Note : Probably less robust and slower that theoretical_V

        """
        pi = self.ode(time=10000)[1][-1]
        print("Calculated fix point for MF")
        print("RMF:")
        print('\r', " Comupting RMF...", end='')
        if order == 0:
            return pi
        if (order >= 1):  # We need 2 derivatives and Q to get the O(1/N)-term
            Q, Fp, Fpp,  = self.defineDriftDerivativeQ(pi)

            # reshaping tensors
            pi = pi.flatten()
            Q = Q.reshape((self.dimensions[0]*self.dimensions[1], self.dimensions[0]*self.dimensions[1]))
            Fp = Fp.reshape((self.dimensions[0]*self.dimensions[1], self.dimensions[0]*self.dimensions[1]))
            Fpp = Fpp.reshape((self.dimensions[0]*self.dimensions[1], self.dimensions[0]*self.dimensions[1],
                               self.dimensions[0]*self.dimensions[1]))

            # reduce dimension for steady state calculation
            print('\r', " Dimension Reduction...", end='')
            Fp, Fpp, Q, P, Pinv, rank = self.reduceDimensionFpFppQ(Fp, Fpp, Q)
            if order == 1:
                pi, V, (V, W) = self.computePiV(pi, Fp, Fpp, Q)
                V, W = self.expandDimensionVW(V, W, Pinv)

                # reshaping into original representation
                pi = pi.reshape(self.dimensions)
                V = V.reshape(self.dimensions)
                W = W.reshape((self.dimensions[0], self.dimensions[1], self.dimensions[0], self.dimensions[1]))
                print('\r', "RMF Fixpoint calculated")
                return pi, V, (V, W)

    def reduceDimensionFpFppQ(self, Fp, Fpp, Q):
        P, P_inv, rank = self.dimensionReduction(Fp)
        Fp = (P@Fp@P_inv)[0:rank, 0:rank]
        Fpp = np.tensordot(np.tensordot(np.tensordot(P, Fpp, 1), P_inv, 1), P_inv,
                    axes=[[1], [0]])[0:rank, 0:rank, 0:rank]
        Q = (P@Q@P.transpose())[0:rank, 0:rank]
        return Fp, Fpp, Q, P, P_inv, rank

    def dimensionReduction(self, A):

        import scipy

        n = len(A)

        # M = np.array([l for l in self._list_of_transitions])
        M = self.get_transition_matrix()
        rank_of_transisions = np.linalg.matrix_rank(M)
        # If rank_of_transisions < n, this means that the stochastic process
        # evolves on a linear subspace of R^n.
        eigenvaluesOfJacobian = scipy.linalg.eig(A, left=False, right=False)
        rank_of_jacobian = np.linalg.matrix_rank(A)
        if sum(np.real(eigenvaluesOfJacobian) < 1e-8) < rank_of_transisions:
            # This means that there are less than "rank_of_transisions"
            # eigenvalues with <0 real part
            print("The Jacobian seems to be not Hurwitz")
        if rank_of_jacobian == n:
            return(np.eye(n), np.eye(n), n)
        C = np.zeros((n, n))
        n = len(A)
        d = 0
        rank_of_previous_submatrix = 0
        for i in range(n):
            rank_of_next_submatrix = np.linalg.matrix_rank(A[0:i+1, 0:i+1])
            if rank_of_next_submatrix > rank_of_previous_submatrix:
                C[d, i] = 1
                d += 1
            rank_of_previous_submatrix = rank_of_next_submatrix
        U, s, V = scipy.linalg.svd(A)
        C[rank_of_jacobian:, :] = U.transpose()[rank_of_jacobian:, :]
        return(C, scipy.linalg.inv(C), rank_of_jacobian)

    def get_transition_matrix(self):
        def e(i,j):
            e = np.zeros(self.dimensions)
            e[i,j] = 1
            return e
        M = []
        # arrivals
        for server in range(self.dimensions[0]):
            for queue_len in range(self.dimensions[1]-1):
                arrival_transition = e(server, queue_len+1) - e(server, queue_len)
                M.append(arrival_transition.flatten())
            for queue_len in range(1, self.dimensions[1]):
                removal_transition = e(server, queue_len-1) - e(server, queue_len)
                M.append(removal_transition.flatten())
        return np.array(M)

    def computePiV(self, pi, Fp, Fpp, Q):
        """ Returns the constants V and W (1/N-term for the steady-state)

        This function assumes that Fp is invertible.
        """
        from scipy.linalg import solve_continuous_lyapunov, inv

        # W = computeW(Fp, Q)
        W = solve_continuous_lyapunov(Fp, -Q)
        # V = computeV(Fp, Fpp, W)
        V = -np.tensordot(inv(Fp),
                          np.tensordot(Fpp, W / 2, 2),
                          1)
        return pi, V, (V, W)

    def expandDimensionVW(self, V, W, Pinv):
        rank = len(V)
        return(Pinv[:, 0:rank]@ V, Pinv[:, 0:rank]@W@Pinv.transpose()[0:rank, :])



if __name__ == "__main__":
    n = 10
    np.random.seed(1)
    # server_rates = (1.4 - 1.) * np.random.random(n) + 1.
    server_rates = np.ones(10)
    server_rates[2:5] = 0.5
    server_rates[5:8] = 3
    arrival_rate = 0.7

    model = lbm(arrival_rate, server_rates)

    pi, V, _ = model.meanFieldExpansionSteadyState()

    # rmf_T, rmf_X, rmf_V = model.expansionTransient(time=30)
    # nr_simulations = 1000
    # sim_T, mean, error1, error2 = model.many_simulations(nr_simulations, steps=5e2)


    # mf_T, mf_X = model.ode(time=250)
    # sim_T, sim_X = model.efficient_simulation(steps=1e6, seed_nr=-1)
    #
    rmf_X_alt = efficient.alternate_state_representation_2(rmf_X)
    rmf_XV_alt = efficient.alternate_state_representation_2(rmf_X + rmf_V)
    # sim_T, sim_X = percentage_state_representation(sim_T, sim_X)

    # sim_T, sim_X = model.efficient_simulation(steps=5e6, seed_nr=-1)
    #
    # time_averages = calc_time_average(sim_X)
    # steady_state_prob_2 = np.sum(time_averages, 0) / model.n
    #
    # print(mf_X[-1][0:steady_state_prob_2.shape[0]] - steady_state_prob_2)

    # for n in [10, 30, 100, 300]:
    #     print(n)
    #     np.random.seed(1)
    #     server_rates = (2.5 - 2.) * np.random.random(n) + 2.0
    #     arrival_rate = 2.
    #     # max_queue_server = [7] * 100
    #
    #     model = lbm(arrival_rate, server_rates)
    #
    #     mf_T, mf_X = model.ode(time=100)
    #     mf_X = alternate_state_representation_2(mf_X)
    #
    #     # time_averages = model.time_average(steps=2e6, seed_nr=-1)
    #     sim_T, sim_X = model.efficient_simulation(steps=1e6, seed_nr=-1)
    #     time_averages = calc_time_average(sim_X)
    #     steady_state_prob_2 = np.sum(time_averages, 0) / model.n
    #     # steady_state_prob_2 = np.sum(time_averages, 0)
    #
    #
    #     max_compare_index = min(steady_state_prob_2.shape[0], mf_X[-1].shape[0])
    #     print((mf_X[-1][0:max_compare_index] - steady_state_prob_2[0:max_compare_index]).max())

    # plotting
    import matplotlib.pyplot as pyplot

    f = pyplot.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    pyplot.subplot(2, 4, 1)
    for i in range(2):
        for j in range(4):
            lower = mean[:, j + (4 * i)] - error2[:, j + (4 * i)]
            upper = mean[:, j + (4 * i)] + error2[:, j + (4 * i)]
            pyplot.subplot(2, 4, j + (4 * i) + 1)
            pyplot.plot(sim_T, mean[:, j + (4 * i)], '-')
            pyplot.plot(rmf_T, rmf_X_alt[:, j + (4 * i)], label='MF')
            pyplot.plot(rmf_T, rmf_XV_alt[:, j + (4 * i)], label='RMF')
            pyplot.fill_between(sim_T, lower, upper, alpha=0.3, label='confidence interval')
            pyplot.ylim([0, 1])
            pyplot.legend()
            pyplot.title("#Server w\ queue len ", str(j + (4 * i)))

    # pyplot.savefig("N{}_S{}.png".format(n))
    pyplot.show()