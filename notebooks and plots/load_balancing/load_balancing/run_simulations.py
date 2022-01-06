import numpy as np

def run_simu_density_representation(models, nr_simulations=3000, nr_steps=None, save_name=''):
    many_sim_T = {}
    many_sim_mean = {}
    many_sim_err1 = {}
    many_sim_err2 = {}

    for n, model in models.items():
        # if n < 30:
        #     nr_simulations = 1000
        #     steps = 4e3
        # elif n >= 30:
        #     nr_simulations = 2000
        #     steps = 7e3
        # elif n >= 50:
        #     nr_simulations = 4000
        #     steps = 9e3
        if nr_steps is None:
            steps = (n / 10) * 2e3
        else:
            steps = nr_steps[n]
        print("Simulation {} Nr of steps of the Process".format(steps))

        many_sim_T[n], many_sim_mean[n], many_sim_err1[n], many_sim_err2[n] = model.many_simulations(
            nr_simulations, steps=steps)

    np.save("saves/" + save_name + "many_sim_mean.npy", many_sim_mean)
    np.save("saves/" + save_name + "many_sim_T.npy", many_sim_T)
    np.save("saves/" + save_name + "many_sim_err1.npy", many_sim_err1)
    np.save("saves/" + save_name + "many_sim_err2.npy", many_sim_err2)

def run_simu_all_states(models, nr_simulations=3000, nr_steps=None, save_name=''):
    many_sim_T_single = {}
    many_sim_mean_single = {}
    many_sim_err1_single = {}
    many_sim_err2_single = {}

    for n, model in models.items():
        # if n < 30:
        #     nr_simulations = 2000
        #     steps = 3e3
        # elif n == 30:
        #     nr_simulations = 3000
        #     steps = 5e3
        # elif n == 50:
        #     nr_simulations = 4000
        #     steps = 7e3
        if nr_steps is None:
            steps = (n / 10) * 2e3
        else:
            steps = nr_steps[n]
        print("Simulation {} Nr of steps of the Process".format(steps))

        print("Nr. items: ", n)
        many_sim_T_single[n], many_sim_mean_single[n], many_sim_err1_single[n], many_sim_err2_single[n] = \
            model.many_simulations_single(nr_simulations, steps=steps)

    np.save("saves/" + save_name + "many_sim_mean_single.npy", many_sim_mean_single)
    np.save("saves/" + save_name + "many_sim_T_single.npy", many_sim_T_single)
    np.save("saves/" + save_name + "many_sim_err1_single.npy", many_sim_err1_single)
    np.save("saves/" + save_name + "many_sim_err2_single.npy", many_sim_err2_single)

def run_simu_average_queue(models, nr_simulations=3000, nr_steps=None, save_name=''):
    many_sim_T = {}
    many_sim_mean_avg_len = {}
    many_sim_err1_avg_len = {}
    many_sim_err2_avg_len = {}

    for n, model in models.items():
        # if n < 30:
        #     nr_simulations = 2000
        #     steps = 4e3
        # elif n >= 30:
        #     nr_simulations = 3000
        #     steps = 7e3
        # elif n >= 50:
        #     nr_simulations = 4000
        #     steps = 9e3

        if nr_steps is None:
            steps = (n / 10) * 2e3
        else:
            steps = nr_steps[n]
        print("Simulation {} Nr of steps of the Process".format(steps))

        many_sim_T[n], many_sim_mean_avg_len[n], many_sim_err1_avg_len[n], many_sim_err2_avg_len[n] \
            = model.many_simulations_avg_queue_len(nr_simulations, steps=steps)

    np.save("saves/" + save_name + "many_sim_mean_avg_len.npy", many_sim_mean_avg_len)
    np.save("saves/" + save_name + "many_sim_T_avg_len.npy", many_sim_T)
    np.save("saves/" + save_name + "many_sim_err1_avg_len.npy", many_sim_err1_avg_len)
    np.save("saves/" + save_name + "many_sim_err2_avg_len.npy", many_sim_err2_avg_len)

if __name__ == "__main__":
    from load_balancing_model import lbm
    import jit_functions as efficient
    import matplotlib.pyplot as plt

    from copy import copy

    from scipy.interpolate import interp1d

    nr_items = [10, 15, 20, 25, 30, 40]
    nr_items = [10, 20, 30, 40]
    max_server_rate = 1.4
    min_server_rate = 1.0
    arrival_rate = 1.0

    np.random.seed(1)
    server_rates = (max_server_rate - min_server_rate) * \
                   np.random.random(size=nr_items[-1]) + min_server_rate

    server_rates[[0, 5, 10, 15, 20, 25, 30, 35]] = 2
    server_rates[[1, 6, 11, 16, 21, 26, 31, 36]] = 0.5

    # 0-percentage representation, 1-single state, 2-avg. queue length
    simulation_type = 2

    models = {}
    comparison_models = {}

    # initialize models
    for n in nr_items:
        models[n] = lbm(arrival_rate, server_rates[:n], verbose=True)
        print('')
        average_server_rate = np.sum(server_rates[:n])
        comparison_models[n] = lbm(arrival_rate, np.ones(shape=(n)) * average_server_rate)

    # Calculate expected value simulation
    if simulation_type == 0:
        many_sim_T = {}
        many_sim_mean = {}
        many_sim_err1 = {}
        many_sim_err2 = {}

        for n, model in models.items():
            if n < 30:
                nr_simulations = 1000
                steps = 4e3
            elif n >= 30:
                nr_simulations = 2000
                steps = 7e3
            elif n >= 50:
                nr_simulations = 4000
                steps = 9e3
            print("Nr. items: ", n)
            many_sim_T[n], many_sim_mean[n], many_sim_err1[n], many_sim_err2[n] = model.many_simulations(
                nr_simulations, steps=steps)

        np.save("saves/many_sim_mean.npy", many_sim_mean)
        np.save("saves/many_sim_T.npy", many_sim_T)
        np.save("saves/many_sim_err1.npy", many_sim_err1)
        np.save("saves/many_sim_err2.npy", many_sim_err2)

    elif simulation_type == 1:
        many_sim_T_single = {}
        many_sim_mean_single = {}
        many_sim_err1_single = {}
        many_sim_err2_single = {}

        for n, model in models.items():
            if n < 30:
                nr_simulations = 2000
                steps = 3e3
            elif n == 30:
                nr_simulations = 3000
                steps = 5e3
            elif n == 50:
                nr_simulations = 4000
                steps = 7e3
            print("Nr. items: ", n)
            many_sim_T_single[n], many_sim_mean_single[n], many_sim_err1_single[n], many_sim_err2_single[n] = \
                model.many_simulations_single(nr_simulations, steps=steps)

        np.save("saves/many_sim_mean_single.npy", many_sim_mean_single)
        np.save("saves/many_sim_T_single.npy", many_sim_T_single)
        np.save("saves/many_sim_err1_single.npy", many_sim_err1_single)
        np.save("saves/many_sim_err2_single.npy", many_sim_err2_single)

    elif simulation_type == 2:
        many_sim_T = {}
        many_sim_mean_avg_len = {}
        many_sim_err1_avg_len = {}
        many_sim_err2_avg_len = {}

        for n, model in models.items():
            if n < 30:
                nr_simulations = 2000
                steps = 4e3
            elif n >= 30:
                nr_simulations = 3000
                steps = 7e3
            elif n >= 50:
                nr_simulations = 4000
                steps = 9e3
            print("Nr. items: ", n)
            many_sim_T[n], many_sim_mean_avg_len[n], many_sim_err1_avg_len[n], many_sim_err2_avg_len[n] \
                = model.many_simulations_avg_queue_len(nr_simulations, steps=steps)

        np.save("saves/many_sim_mean_avg_len.npy", many_sim_mean_avg_len)
        np.save("saves/many_sim_T_avg_len.npy", many_sim_T)
        np.save("saves/many_sim_err1_avg_len.npy", many_sim_err1_avg_len)
        np.save("saves/many_sim_err2_avg_len.npy", many_sim_err2_avg_len)
