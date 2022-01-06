from numba import jit, njit
import numba
import numpy as np
import random as rnd


@jit(nopython=True)
def simulate_jit(x0, arrival_rate, server_rates, steps=1e6, seed_nr=-1, verbose=False):
    t = 0.

    n = int(x0.shape[0])
    x = x0
    time_index = 0
    max_steps = int(steps)
    # initialize vectors
    s = np.zeros(shape=30, dtype=np.float64)
    s[0] = n
    times = np.empty(shape=max_steps, dtype=np.float64)
    states = np.empty(shape=(max_steps, n), dtype=np.float64)
    times[time_index] = t
    states[time_index] = x

    if seed_nr != -1:
        np.random.seed(seed_nr)

    service_rate_cumsum = np.cumsum(np.asarray(server_rates))
    total_server_rates = service_rate_cumsum[-1]
    service_rate_cumsum /= total_server_rates

    time_index += 1
    # simulate the markov chain
    while time_index < max_steps:
        rates_sum = arrival_rate * n + total_server_rates

        # transition selection
        a = np.random.random()
        adding_discarding = arrival_rate * n / rates_sum
        if a < adding_discarding: # this is an arrival of a job
            # add a job to one queue : we pick the shortest among two
            idx1 = np.random.randint(n)
            idx2 = np.random.randint(n)
            # Set index to the bigger one of the servers.
            index = idx1 if x[idx1] <= x[idx2] else idx2
            s[int(x[index]) + 1] += 1
            x[index] += 1.
        else:
            # discard a job : we choose a random queue
            a = np.random.random()
            index = np.searchsorted(service_rate_cumsum, a, side="right")
            if x[index] > 0: # there is a departure only if x[index] > 0
                s[int(x[index])] -= 1
                x[index] -= 1.

        t += np.random.exponential(1/rates_sum)


        # if verbose and time_index % 100000 == 0:
        #     print(time_index)

        times[time_index] = t
        states[time_index] = x

        time_index += 1

    return times, states

@jit(nopython=True)
def simulate_jit_time_spend(x0, arrival_rate, server_rates, steps=1e6, seed_nr=-1, verbose=False):
    t = 0.

    n = x0.shape[0]
    x = x0
    time_index = 0
    max_steps = int(steps)
    # initialize vectors
    s = np.zeros(shape=30, dtype=np.float64)
    s[0] = n
    times = np.empty(shape=max_steps, dtype=np.float64)
    states_1 = np.empty(shape=(n, 20), dtype=np.float64)
    times[time_index] = t
    states_1[time_index] = x

    if seed_nr != -1:
        np.random.seed(seed_nr)

    service_rate_cumsum = np.cumsum(np.asarray(server_rates))
    total_server_rates = service_rate_cumsum[-1]
    service_rate_cumsum /= total_server_rates


    # simulate the markov chain
    while time_index < max_steps:
        rates_sum = arrival_rate * n + total_server_rates
        if rates_sum <= 1e-14:
            print("System stalled -> total transitions rate = 0")
        else:
            # transition selection
            a = np.random.random()
            adding_discarding = arrival_rate * n / rates_sum
            if a < adding_discarding: # this is an arrival of a job
                # add a job to one queue : we pick the shortest among two
                idx1 = np.random.randint(n)
                idx2 = np.random.randint(n)
                # Set index to the bigger one of the servers.
                index = idx1 if x[idx1] <= x[idx2] else idx2
                s[int(x[index]) + 1] += 1
                x[index] += 1.
            else:
                # discard a job : we choose a random queue
                a = np.random.random()
                index = np.searchsorted(service_rate_cumsum, a, side="right")
                if x[index]>0: # there is a departure only if x[index] > 0
                    s[int(x[index])] -= 1
                    x[index] -= 1.

        t += np.random.exponential(1/rates_sum)
        time_index += 1

        # if verbose and time_index % 100000 == 0:
        #     print(time_index)

        times[time_index] = t
        states_1[time_index] = x

    return times, states_1

@jit(nopython=True)
def get_arrival(x, s):
    # Todo: pass and update s instead of calculating it new every time
    n = x.shape[0]
    a = rnd.random()

    prob_3 = np.zeros(shape=n)

    for i in range(n):
        prob_3[i] = (s[int(x[i])] + s[int(x[i])+1])/(n**2)
    index = np.searchsorted(np.cumsum(prob_3), a, side="right")
    return index

    return np.random.choice(np.where(x==index)[0], 1)

        # probabilities[queue_size] += (np.sum(x >= queue_size) + np.sum(x >= queue_size+1))/n

@jit(nopython=True)
def get_removal(x, server_rates):
    probabilities = (server_rates / np.sum(server_rates))
    a = rnd.random()
    index = np.searchsorted(np.cumsum(probabilities), a, side="right")
    return index

@jit(nopython=True)
def testcomparisonsimulation(x0, arrival_rate, server_rates, steps=1e6, seed_nr=-1, console_output=False):
    t = 0.

    n = x0.shape[0]
    x = x0
    time_index = 0
    max_steps = int(steps)
    # initialize vectors
    s = np.zeros(shape=30, dtype=np.float64)
    s[0] = n
    times = np.empty(shape=max_steps, dtype=np.float64)
    states = np.empty(shape=(max_steps, n), dtype=np.float64)
    times[time_index] = t
    states[time_index] = x

    if seed_nr != -1:
        rnd.seed(seed_nr)

    # simulate the markov chain
    while time_index < max_steps:

        active_server_rates = 0
        for i in range(x.shape[0]):
            if x[i] > 0:
                active_server_rates += server_rates[i]
        rates_sum =  arrival_rate * n + active_server_rates
        if rates_sum <= 1e-14:
            print("System stalled -> total transitions rate = 0")
            # t = time
        else:
            # transition selection
            a = rnd.random()
            adding_discarding = arrival_rate * n / rates_sum
            if a < adding_discarding:
                # add a job to one queue
                index = get_arrival(x, s)
                s[int(x[index]) + 1] += 1
                x[index] += 1.

            else:
                # discard a job
                index = get_removal(x, np.asarray([server_rates[i] if x[i] > 0 else 0 for i in range(x.shape[0])],dtype=np.float64))
                s[int(x[index])] -= 1
                x[index] -= 1.

            t += rnd.expovariate(rates_sum)
            time_index +=1

            if console_output and time_index % 100000 == 0:
                print(time_index)

        times[time_index] = t
        states[time_index] = x

    return times, states


@jit(nopython=True)
def percentage_state_representation(T, X, q_max=12):
    # d2_max - max number of items in queue
    # if T[-1] != 0:
    nr_steps = T.shape[0]
    # else:
    #     # get first entry where time array is zero (excluding the initial state)
    #     nr_steps = int(np.where(T[1:] == 0)[0][0] + 1.)

    new_T = T[:nr_steps]
    new_X = np.zeros(shape=(nr_steps, q_max), dtype=np.float64)
    for _t in range(nr_steps):
        for i in range(q_max):
            new_X[_t, i] = np.sum(X[_t] == i) / X.shape[1]
    return new_T, new_X

@jit(nopython=True)
def percentage_state_representation_single(T, X, q_max=12):
    # d2_max - max number of items in queue
    # if T[-1] != 0:
    n = X.shape[1]
    nr_steps = T.shape[0]
    # else:
    #     # get first entry where time array is zero (excluding the initial state)
    #     nr_steps = int(np.where(T[1:] == 0)[0][0] + 1.)

    new_T = T[:nr_steps]
    new_X = np.zeros(shape=(nr_steps, n, q_max), dtype=np.float64)
    for _t in range(nr_steps):
        for i in range(q_max):
            new_X[_t, :, i] = (X[_t] == i)
    return new_T, new_X

@jit(nopython=True)
def alternate_state_representation_2(state_array):
    def transform_state(state):
        # get array displaying number of servers with equal queue lengths
        max_queue_length = state.shape[1]
        n = state.shape[0]
        new_state = np.zeros(shape=(max_queue_length), dtype=np.float64)
        for i in range(max_queue_length):
            # for each queue length sum over servers
            new_state[i] = np.sum(state[:, i])
        return new_state / n


    # return a (time-)series of states
    new_state_array = np.zeros(shape=(state_array.shape[0], state_array.shape[2]), dtype=np.float64)
    for state in range(state_array.shape[0]):
        new_state_array[state] = transform_state(state_array[state])
    return new_state_array

@jit(nopython=True)
def efficient_drift(x, arrival_rate, mu):
    n = x.shape[0]
    q_len = x.shape[1]
    _drift_vec = np.zeros(shape=x.shape, dtype=np.float64)
    _s = np.zeros(shape=x.shape[1], dtype=np.float64)

    for j in range(q_len):
        # calculate in reverse order or subtract
        _s[j] = np.sum(x[:, j:])

    for j in range(1, q_len, 1):
        transition = np.zeros(shape=x.shape, dtype=np.float64)

        rate = (x[:, j - 1] * (_s[j - 1] + _s[j])) / n
        rate *= arrival_rate
        transition[:, j] += rate
        transition[:, j - 1] += - rate

        _drift_vec += transition
        # test
        # _drift_vec[:,j] += rate
        # _drift_vec[:,j - 1] += -rate
    for j in range(0, q_len - 1, 1):
        transition = np.zeros(shape=x.shape, dtype=np.float64)

        rate = x[:, j + 1] * mu[:]
        # rate = x[server, j+1] * mu[server]
        transition[:, j] += rate
        transition[:, j + 1] += - rate

        _drift_vec += transition
        # _drift_vec[:,j] += rate
        # _drift_vec[:,j + 1] += -rate

    return _drift_vec

@jit(nopython=True)
def drift2(x, arrival_rate, mu):
    """
    Alternative drift (probably less efficient)
    """
    n = x.shape[0]
    max_q = x.shape[1]
    _drift = np.zeros(shape=x.shape, dtype=np.float64)

    for server1 in range(n):
        # adding a job
        for server2 in range(n):
            for s1 in range(0, max_q -1):
                for s2 in range(0, max_q -1):
                    rate = arrival_rate * x[server1, s1]*x[server2, s2] / n
                    if s1 <= s2:
                        _drift[server1, s1]   -= rate
                        _drift[server1, s1+1] += rate
                    else:
                        _drift[server2, s2]   -= rate
                        _drift[server2, s2+1] += rate
    for server in range(n):
        for j in range(1, max_q):
            rate = x[server, j] * mu[server]
            _drift[server, j]   -= rate
            _drift[server, j-1] += rate
    return _drift

@jit(nopython=True)
def jacobian(x, arrival_rate, mu):
    n = x.shape[0]
    q_len = x.shape[1]
    _d_drift = np.zeros(shape=(n, q_len, n, q_len), dtype=np.float64)

    for server1 in range(n):
        # adding a job
        for server2 in range(n):
            for s1 in range(0, q_len):
                for s2 in range(0, q_len):
                    for i0, k0, i1, k1 in [(server1, s1, server2, s2), (server2, s2, server1, s1)]:
                        rate = x[i0, k0]
                        # rate = arrival_rate * x[i0, k0] / n
                        if s1 <= s2 and s1 <= q_len-2:
                            _d_drift[server1, s1, i1, k1] -= rate
                            _d_drift[server1, s1 + 1, i1, k1] += rate
                        elif s2 <= q_len-2:
                            _d_drift[server2, s2, i1, k1] -= rate
                            _d_drift[server2, s2 + 1, i1, k1] += rate
    _d_drift *= arrival_rate / n
    for server in range(n):
        rate = mu[server]
        for j in range(1, q_len):
            _d_drift[server, j, server, j] -= rate
            _d_drift[server, j - 1, server, j] += rate
    return _d_drift

@jit(nopython=True)
def hessian(x, arrival_rate):
    n = x.shape[0]
    q_len = x.shape[1]
    _dd_drift = np.zeros(shape=(n, q_len, n, q_len, n, q_len), dtype=np.float64)

    rate = arrival_rate / n

    for server1 in range(n):
        # adding a job
        for server2 in range(n):
            for s1 in range(0, q_len):
                for s2 in range(0, q_len):
                    for i0, k0, i1, k1 in [(server1, s1, server2, s2), (server2, s2, server1, s1)]:
                        if s1 <= s2 and s1 <= q_len-2:
                            _dd_drift[server1, s1, i1, k1, i0, k0] -= rate
                            _dd_drift[server1, s1 + 1, i1, k1, i0, k0] += rate
                        elif s2 <= q_len-2:
                            _dd_drift[server2, s2, i1, k1, i0, k0] -= rate
                            _dd_drift[server2, s2 + 1, i1, k1, i0, k0] += rate
    return _dd_drift

@jit(nopython=True)
def qMatrix(x, arrival_rate, mu):
    n = x.shape[0]
    q_len = x.shape[1]
    _q = np.zeros(shape=(n, q_len, n, q_len), dtype=np.float64)

    for server1 in range(n):
        # adding a job
        for server2 in range(n):
            for s1 in range(0, q_len):
                for s2 in range(0, q_len):
                    rate = arrival_rate * x[server1, s1] * x[server2, s2] / n
                    if s1 <= s2 and s1 <= q_len-2:
                        _q[server1, s1, server1, s1] += rate
                        _q[server1, s1 + 1, server1, s1 + 1] += rate
                        _q[server1, s1 + 1, server1, s1] -= rate
                        _q[server1, s1, server1, s1 + 1] -= rate
                    elif s2 <= q_len-2:
                        _q[server2, s2, server2, s2] += rate
                        _q[server2, s2 + 1, server2, s2 + 1] += rate
                        _q[server2, s2 + 1, server2, s2] -= rate
                        _q[server2, s2, server2, s2 + 1] -= rate
    for server1 in range(n):
        for s1 in range(1, q_len):
            rate = x[server1, s1] * mu[server1]
            _q[server1, s1, server1, s1] += rate
            _q[server1, s1 - 1, server1, s1 - 1] += rate
            _q[server1, s1, server1, s1 - 1] -= rate
            _q[server1, s1 - 1, server1, s1] -= rate
    return _q

@jit(nopython=True)
def qMatrix(x, arrival_rate, mu):
    n = x.shape[0]
    q_len = x.shape[1]
    _q = np.zeros(shape=(n, q_len, n, q_len), dtype=np.float64)

    for server1 in range(n):
        # adding a job
        for server2 in range(n):
            for s1 in range(0, q_len - 1):
                for s2 in range(0, q_len - 1):
                    rate = arrival_rate * x[server1, s1] * x[server2, s2] / n
                    if s1 <= s2:
                        _q[server1, s1, server1, s1] += rate
                        _q[server1, s1 + 1, server1, s1 + 1] += rate
                        _q[server1, s1 + 1, server1, s1] -= rate
                        _q[server1, s1, server1, s1 + 1] -= rate
                    else:
                        _q[server2, s2, server2, s2] += rate
                        _q[server2, s2 + 1, server2, s2 + 1] += rate
                        _q[server2, s2 + 1, server2, s2] -= rate
                        _q[server2, s2, server2, s2 + 1] -= rate

    for server1 in range(n):
        for s1 in range(1, q_len):
            rate = x[server1, s1] * mu[server1]
            _q[server1, s1, server1, s1] += rate
            _q[server1, s1 - 1, server1, s1 - 1] += rate
            _q[server1, s1, server1, s1 - 1] -= rate
            _q[server1, s1 - 1, server1, s1] -= rate
    return _q


# @jit(nopython=True)
def calc_time_average(x):
    start_index = int(5e5)
    total_time = x[start_index:].shape[0]

    max_q_size = int(x.max())+1
    time_in_state = np.zeros(shape=(x.shape[1], max_q_size), dtype=np.float64)
    for i in range(max_q_size):
        time_in_state[:, i] = np.sum(x[start_index:] == i, axis=0)

    normalized_rep = time_in_state / total_time
    return normalized_rep

@jit(nopython=True)
def calc_time_average_2(x):
    start_index = int(1e6)
    total_time = x[start_index:].shape[0]

    max_q_size = int(x.max())+1
    time_in_state = np.zeros(shape=(x.shape[1], max_q_size), dtype=np.float64)
    factor = 1 / total_time
    servers = np.arange(0,x.shape[1])
    for t in range(start_index, start_index + total_time):
        for i in range(x[t].shape[0]):
            time_in_state[servers[i], int(x[t,i])] = time_in_state[servers[i], int(x[t,i])] + factor

    return time_in_state



if __name__ == "__main__":
    import load_balancing_model
    from copy import copy
    nr_items = [50]  # , 70]#, 100] [10, 15, 30,
    max_server_rate = 1.4
    min_server_rate = 1.0
    arrival_rate = 1.0

    np.random.seed(1)
    server_rates = (max_server_rate - min_server_rate) * \
                   np.random.random(size=nr_items[-1]) + min_server_rate
    print(server_rates)
    models = {}

    # initialize models
    for n in nr_items:
        models[n] = load_balancing_model.lbm(arrival_rate, server_rates[:n])

    results_mf = []
    results_rmf = []

    steady_state_prob_all_simu = {}
    steady_state_prob_all_rmf = {}
    steady_state_prob_all_mf = {}

    trans_state_X = {}
    trans_state_XV = {}
    trans_state_T = {}

    # Run approximations (MF / RMF) and steady state simulation
    for n, model in models.items():
        print("Calculating results for n={}.".format(n))

        # rmf_T, rmf_X, rmf_V = model.expansionTransient(time=50)
        # mf_X = alternate_state_representation_2(rmf_X)
        # rmf_V = alternate_state_representation_2(rmf_V)
        # rmf_XV = efficient.alternate_state_representation_2(rmf_X + rmf_V)
        # rmf_XV = mf_X + rmf_V
        # trans_state_X[n] = copy(mf_X)
        # trans_state_XV[n] = copy(rmf_XV)
        # trans_state_T[n] = copy(rmf_T)

        print("Simulating")
        sim_T, sim_X = model.efficient_simulation(steps=1e7, seed_nr=-1)
        print("Calculating Time averages")
        time_averages = calc_time_average_2(sim_X)
        steady_state_prob = np.sum(time_averages, axis=0) / model.n
        steady_state_prob_all_simu[n] = steady_state_prob

        # steady state approximation
        print("Steady state")
        pi, V, _ = model.meanFieldExpansionSteadyState()
        pi_rmf = pi + V

        pi = alternate_state_representation_2(np.array([pi]))[0]
        pi_rmf = alternate_state_representation_2(np.array([pi + V]))[0]

        # gives max range of indices which can be compared
        # max_compare_index = min(steady_state_prob.shape[0], mf_X[-1].shape[0])
        # max_abs_difference_mf = np.abs(mf_X[-1][0:max_compare_index] - steady_state_prob[0:max_compare_index]).max()
        # results_mf.append(max_abs_difference_mf)
        # steady_state_prob_all_mf[n] = mf_X[-1]

        max_compare_index = min(steady_state_prob.shape[0], pi.shape[0])
        max_abs_difference_mf = np.abs(pi[0:max_compare_index] - steady_state_prob[0:max_compare_index]).max()
        results_mf.append(max_abs_difference_mf)
        steady_state_prob_all_mf[n] = copy(pi)
        print(max_abs_difference_mf)

        # max_compare_index = min(steady_state_prob.shape[0], rmf_XV[-1].shape[0])
        # max_abs_difference_rmf = np.abs(rmf_XV[-1][0:max_compare_index] - steady_state_prob[0:max_compare_index]).max()
        # results_rmf.append(max_abs_difference_rmf)
        # steady_state_prob_all_rmf[n] = rmf_XV[-1]

        max_compare_index = min(steady_state_prob.shape[0], pi_rmf.shape[0])
        max_abs_difference_rmf = np.abs(pi_rmf[0:max_compare_index] - steady_state_prob[0:max_compare_index]).max()
        results_rmf.append(max_abs_difference_rmf)
        steady_state_prob_all_rmf[n] = copy(pi_rmf)
        print(max_abs_difference_rmf)

    # save results for steady state and transient state
    np.savetxt("saves/steady_state_comparison_mf.txt", np.array([nr_items, results_mf]))
    np.savetxt("saves/steady_state_comparison_rmf.txt", np.array([nr_items, results_rmf]))

    np.save("saves/steady_state_prob_all_mf.npy", steady_state_prob_all_mf)
    np.save("saves/steady_state_prob_all_rmf.npy", steady_state_prob_all_rmf)
    np.save("saves/steady_state_prob_all_simu.npy", steady_state_prob_all_simu)

    # np.save("trans_state_X.npy", trans_state_X)
    # np.save("trans_state_XV.npy", trans_state_XV)
    # np.save("trans_state_T.npy", trans_state_T)






