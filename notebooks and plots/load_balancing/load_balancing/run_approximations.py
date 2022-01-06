from load_balancing.load_balancing_model import lbm
import load_balancing.jit_functions as efficient
import matplotlib.pyplot as plt

from copy import copy

import numpy as np
from scipy.interpolate import interp1d

def run_simu_all_states(models, save_file='', hom_model=False):
    results_mf = []
    results_rmf = []

    steady_state_prob_all_simu = {}
    steady_state_prob_all_rmf = {}
    steady_state_prob_all_mf = {}

    trans_state_X = {}
    trans_state_XV = {}
    trans_state_T = {}

    single_trans_state_X = {}
    single_trans_state_XV = {}

    single_steady_state_X = {}
    single_steady_state_XV = {}


    # Run approximations (MF / RMF) and steady state simulation
    nr_items = list(models.keys())
    print(nr_items)

    for n, model in models.items():

        print("Calculating results for n={}.".format(n))

        # rmf_T, rmf_X, rmf_V = model.expansionTransient(time=80)
        # single_trans_state_X[n] = copy(rmf_X)
        # single_trans_state_XV[n] = copy(rmf_X + rmf_V)
        # mf_X = efficient.alternate_state_representation_2(rmf_X)
        # rmf_V = efficient.alternate_state_representation_2(rmf_V)
        # rmf_XV = efficient.alternate_state_representation_2(rmf_X + rmf_V)
        # rmf_XV = mf_X + rmf_V
        # trans_state_X[n] = copy(mf_X)
        # trans_state_XV[n] = copy(rmf_XV)
        # trans_state_T[n] = copy(rmf_T)

        if not hom_model:
            print("Simulating heterogeneous model")
            if n == 5:
                sim_T, sim_X = models[n].efficient_simulation(steps=2e7, seed_nr=-1)
            else:
                sim_T, sim_X = models[n].efficient_simulation(steps=1e7, seed_nr=-1)
            print("Calculating Time averages")
            time_averages = efficient.calc_time_average_2(sim_X)
            steady_state_prob = np.sum(time_averages, axis=0) / n
            steady_state_prob_all_simu[n] = steady_state_prob

        # steady state approximation
        pi, V, _ = model.meanFieldExpansionSteadyState()
        single_steady_state_X[n] = copy(pi)
        single_steady_state_XV[n] = copy(pi + V)

        pi = efficient.alternate_state_representation_2(np.array([pi]))[0]
        pi_rmf = efficient.alternate_state_representation_2(np.array([pi + V]))[0]

        # gives max range of indices which can be compared
        # max_compare_index = min(steady_state_prob.shape[0], mf_X[-1].shape[0])
        # max_abs_difference_mf = np.abs(mf_X[-1][0:max_compare_index] - steady_state_prob[0:max_compare_index]).max()
        # results_mf.append(max_abs_difference_mf)
        # steady_state_prob_all_mf[n] = mf_X[-1]
        if not hom_model:
            max_compare_index = min(steady_state_prob.shape[0], pi.shape[0])
            max_abs_difference_mf = np.abs(pi[0:max_compare_index] - steady_state_prob[0:max_compare_index]).max()
            results_mf.append(max_abs_difference_mf)
            print(max_abs_difference_mf)

        steady_state_prob_all_mf[n] = copy(pi)


        # max_compare_index = min(steady_state_prob.shape[0], rmf_XV[-1].shape[0])
        # max_abs_difference_rmf = np.abs(rmf_XV[-1][0:max_compare_index] - steady_state_prob[0:max_compare_index]).max()
        # results_rmf.append(max_abs_difference_rmf)
        # steady_state_prob_all_rmf[n] = rmf_XV[-1]

        if not hom_model:
            max_compare_index = min(steady_state_prob.shape[0], pi_rmf.shape[0])
            max_abs_difference_rmf = np.abs(pi_rmf[0:max_compare_index] - steady_state_prob[0:max_compare_index]).max()
            results_rmf.append(max_abs_difference_rmf)
            print(max_abs_difference_rmf)

        steady_state_prob_all_rmf[n] = copy(pi_rmf)


    # save results for steady state and transient state
    if hom_model:
        # np.savetxt("saves/" + save_file + "comp_steady_state_comparison_mf.txt", np.array([nr_items, results_mf]))
        # np.savetxt("saves/" + save_file + "comp_steady_state_comparison_rmf.txt", np.array([nr_items, results_rmf]))

        np.save("saves/" + save_file + "comp_steady_state_prob_all_mf.npy", steady_state_prob_all_mf)
        np.save("saves/" + save_file + "comp_steady_state_prob_all_rmf.npy", steady_state_prob_all_rmf)
        # np.save("saves/" + save_file + "comp_steady_state_prob_all_simu.npy", steady_state_prob_all_simu)

        # np.save("saves/" + save_file + "comp_trans_state_X.npy", trans_state_X)
        # np.save("saves/" + save_file + "comp_trans_state_XV.npy", trans_state_XV)
        # np.save("saves/" + save_file + "comp_trans_state_T.npy", trans_state_T)

        # np.save("saves/" + save_file + "comp_single_trans_state_X.npy", single_trans_state_X)
        # np.save("saves/" + save_file + "comp_single_trans_state_XV.npy", single_trans_state_XV)

        np.save("saves/" + save_file + "comp_single_steady_state_X.npy", single_steady_state_X)
        np.save("saves/" + save_file + "comp_single_steady_state_X.npy", single_steady_state_XV)
    else:
        np.savetxt("saves/" + save_file + "steady_state_comparison_mf.txt", np.array([nr_items, results_mf]))
        np.savetxt("saves/" + save_file + "steady_state_comparison_rmf.txt", np.array([nr_items, results_rmf]))

        np.save("saves/" + save_file + "steady_state_prob_all_mf.npy", steady_state_prob_all_mf)
        np.save("saves/" + save_file + "steady_state_prob_all_rmf.npy", steady_state_prob_all_rmf)
        np.save("saves/" + save_file + "steady_state_prob_all_simu.npy", steady_state_prob_all_simu)

        np.save("saves/" + save_file + "trans_state_X.npy", trans_state_X)
        np.save("saves/" + save_file + "trans_state_XV.npy", trans_state_XV)
        np.save("saves/" + save_file + "trans_state_T.npy", trans_state_T)

        np.save("saves/" + save_file + "single_trans_state_X.npy", single_trans_state_X)
        np.save("saves/" + save_file + "single_trans_state_XV.npy", single_trans_state_XV)

        np.save("saves/" + save_file + "single_steady_state_X.npy", single_steady_state_X)
        np.save("saves/" + save_file + "single_steady_state_X.npy", single_steady_state_XV)


if __name__ == "__main__":
    # system sizes
    nr_items = [40, 30, 25, 20, 10, 15]

    arrival_rate = 1.0

    # set server rates
    ## uniformly distributed servers
    max_server_rate = 1.5
    min_server_rate = 1.0

    np.random.seed(1)
    server_rates = (max_server_rate - min_server_rate) * \
                   np.random.random(size=max(nr_items)) + min_server_rate

    print(server_rates)

    # set all servers to one (other option)
    # server_rates = np.ones(nr_items[-1])

    # set specific server speeds (here 20 percent have value 2 and another 20 percent 0.5)
    # for clarification: server_rates[0:10] corresponds to the first 10 servers the servers of the first model of size 10
    # server_rates[0:20] to the second model  with size 20, etc.
    # server_rates[[0, 5, 10, 15, 20, 25, 30, 35]] = 2
    # server_rates[[1, 6, 11, 16, 21, 26, 31, 36]] = 0.5

    models = {}
    comparison_models = {}
    simulation_steps = {}

    # initialize dictionary with models based on the specifications
    for n in nr_items:
        _server_rates = server_rates[:n]
        models[n] = lbm(arrival_rate, _server_rates, verbose=True)
        print('')
        average_server_rate = np.sum(server_rates[:n]) / n
        print(average_server_rate)
        print(server_rates[:n])
        comparison_models[n] = lbm(arrival_rate, np.ones(shape=(n)) * average_server_rate, verbose=True)
        if n < 30:
            simulation_steps[n] = 3e3
        elif n >= 30 and n < 50:
            simulation_steps[n] = 5e3
        elif n == 50:
            simulation_steps[n] = 7e3

    run_simu_all_states(models, save_file='uniform_servers', hom_model=False)
    run_simu_all_states(comparison_models, save_file='uniform_servers', hom_model=True)

