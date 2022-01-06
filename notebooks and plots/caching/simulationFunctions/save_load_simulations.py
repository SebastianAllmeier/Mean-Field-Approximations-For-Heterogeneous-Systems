### generate batches of simulationFunctions


import tools.analysis_reproduction
from tools.analysis_reproduction import timer

import numpy as np

import h5py

from datetime import datetime
from cacheModel import initialize_model

def run_and_save_simulations(nr_simulations, nr_items, zipf_alpha, cache_sizes, simulation_time, extra_id=None):
    """
    This function runs and saves simulationFunctions of the RandCache model with adjustable parameters.
    The simulationFunctions are stored in a h5 file. The file stores the initial parameters and each individual simulationFunctions
    with corresponding seed.

    The simulationFunctions will be saved in 'cache/'. The folder has to be created manually.

    Expected parameters
    :param nr_simulations:  [Int] Number of Simulations that should be run.
    :param nr_items:        [Int] Number of items in the model.
    :param zipf_alpha:      [Float] \alpha for the used Zipf distribution.
    :param cache_sizes:     [(Int) Array] Array of cache sizes
    :param simulation_time: [Float] max. time frame for simulationFunctions
    :param extra_id:        [String] (optional) Some additional identifier [string]

    :return: None
    """
    import cacheModel
    # if seed_rates is None:
    #     seed_rates = np.random.randint(0,10000)

    # get random list of seed for simulationFunctions
    simulation_seeds = np.random.randint(0,100000, nr_simulations)


    ## Setup for the model
    # set initial state
    print("Initializing Model")
    print("Parameters:\n\tNr. Simulations: {}\n\tNr. Items: {}\n\tZipf Alpha: {}\n\tCache Size: {}\n\t".format(
        nr_simulations, nr_items, zipf_alpha, cache_sizes) + "Simulations time: {}".format(simulation_time))

    number_of_lists = len(cache_sizes)
    model_dimension = nr_items * (number_of_lists + 1)
    initial_state = np.zeros(model_dimension)
    # we put the first k element in list 1, the next in list 2, etc.
    object_index = nr_items - 1
    initial_obj = np.arange(nr_items)
    # np.random.shuffle(initial_obj)
    # the above line shuffles which elements should be in the cache
    # otherwise the cache are filled in increasing order

    def index(_i, _k, _nr_items):
        return _i + _k * _nr_items

    # specify initial state
    for _list in range(number_of_lists, 0, -1):
        for i in range(cache_sizes[_list-1]):
            initial_state[index(initial_obj[object_index], _list, nr_items)] = 1
            object_index -= 1
    # initialize non cached items
    for i in range(nr_items - sum(cache_sizes)):
        initial_state[index(initial_obj[object_index], 0, nr_items)] = 1
        object_index -= 1

    model = initialize_model(nr_items=nr_items, zipf_alpha=zipf_alpha,
                             cache_sizes=cache_sizes, initial_state=initial_state)
    # # set / load rate parameters
    # zipf_distribution = cacheModel.zipf(n=nr_items, alpha=zipf_alpha)
    #
    # # initialize model with parameters and inital state
    # model = cacheModel.cacheRANDmDDPP(zipf_distribution, cache_sizes)
    # model.set_initial_state(initial_state)
    # # model.define_transitions_and_rates()

    # save seeds, simulationFunctions, rate parameters
    # set name for save file
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = "nr_simulations-" + str(nr_simulations) + "_nr_items-" + str(nr_items) + "_alpha-" + str(zipf_alpha) + "_simulation_time-" + str(simulation_time) + "_" + date
    if extra_id is not None:
        name += "_" + str(extra_id)

    # initialize hdf5 file
    f = h5py.File('cache/' + name + '.h5', 'w')

    for i, seed in enumerate(simulation_seeds):
        T_ctmc, X_ctmc = model.simulate(time=simulation_time, N=1, seed_nr=seed)
        gr = f.create_group('sim_' + str(i))
        gr.create_dataset('times', data=T_ctmc,  compression="gzip", compression_opts=9)
        gr.create_dataset('data', data=X_ctmc, compression="gzip", compression_opts=9)
        gr.create_dataset('seed', data=seed)
        if i % 100 == 0:
            if i == 0:
                print("Running Simulations")
            else:
                print("\t- " + str(i) + " simulationFunctions done.")

    gr = f.create_group('init')
    gr.create_dataset('nr_simulations', data=nr_simulations)
    gr.create_dataset('nr_items', data=nr_items)
    # gr.create_dataset('percentage_infected', data=percentage_infected)
    # gr.create_dataset('simulation_time', data=simulation_time)
    # gr.create_dataset('R_to_S_rates', data=R_to_S_rates)
    # gr.create_dataset('seed_rates', data=seed_rates)
    # gr.create_dataset('rate_parameters', data=[alpha, beta, gamma, delta], compression="gzip", compression_opts=9)
    gr.create_dataset('zipf_alpha', data=zipf_alpha)
    gr.create_dataset('initial_state', data=initial_state)
    gr.create_dataset('simulation_time', data=simulation_time)
    gr.create_dataset('cache_sizes', data=cache_sizes)



    f.close()
    print("Simulations have been saved to " + 'cache/' + name + '.h5' + " .")

def load_simulations(file_path, output=False):
    """
    Opens an existing hdf5 dataset for further processing.

    :param file_path:   [String] Path to the .h5 file which should be loaded
    :param output:      [Bool] print initial parameters of the model for the simulations
    :return:            [HDF5 dataset] opened hdf5 dataset
    """
    f = h5py.File(file_path, 'r')
    initial_values_dict = dict()
    if output:
        print("\nLoaded " + str(file_path))
        print("\nModel initialization parameter and seeds available: \n")
    init_group = f.get("init")

    for key in init_group.keys():
        if output:
            print(str(key) + " :")
            print(np.array(init_group.get(key)))
            print("--------------------")
        initial_values_dict[str(key)] = np.array(init_group.get(key))
    # random_sim_nr = np.random.randint(0,np.array(init_group.get("nr_simulations")))
    # sim_gr = (f.get("sim_" + str(random_sim_nr)))
    # times = np.array(sim_gr.get("times"))
    # data = np.array(sim_gr.get("data"))
    # seed = np.array(sim_gr.get("seed"))
    return f, initial_values_dict

def safe_mean_var_mf(file_path, output=False):
    from tools.analysis_reproduction import save_load
    if output:
        print("Saving lin sum, squared sum, (refined) mean field approx for\n\t" + file_path)
    from simulationFunctions.simulationTools import calculate_lin_square_sum

    # load file
    load_f = h5py.File(file_path, 'r')
    init_group = load_f.get("init")

    # get initial parameters
    nr_items = int(np.array(init_group.get('nr_items')))
    zipf_alpha = float(np.array(init_group.get('zipf_alpha')))
    simulation_time = int(np.array(init_group.get('simulation_time')))
    cache_sizes = np.array(init_group.get('cache_sizes'))
    initial_state = np.array(init_group.get('initial_state'))

    # calculate values
    if output:
        print("Calculating lin and squared terms")
    _lin_sum, _square_sum, _square_sum_hit_rate_caches, _times, _nr_simulations = calculate_lin_square_sum(load_f, time_steps=400)
    # _sim_mean_values, _times, _nr_items = save_load(calculate_sim_mean)(load_f, normalize=False)
    # _sim_var_values, _, _ = save_load(calculate_sim_var)(load_f, normalize=False)

    model = initialize_model(nr_items=nr_items, zipf_alpha=zipf_alpha,
                             cache_sizes=cache_sizes, initial_state=initial_state)

    if output:
        print("Calculating MF and RMF")
    rmf_times, mf_values, rmf_correction, _ = model.meanFieldExpansionTransient(order=1, time=simulation_time)

    # set filename
    cache_sizes_sting = ""
    for cache in cache_sizes:
        cache_sizes_sting += str(cache) + "-"
    cache_sizes_sting = cache_sizes_sting[:-1]

    name = "sim_mean_var__nr_items-" + str(nr_items) + "_zipf_alpha-" + str(zipf_alpha) \
           + "_simulation_time-" + str(simulation_time) + "_cache_sizes-" + cache_sizes_sting

    save_f = h5py.File('cache/' + name + '.h5', 'w')

    # save values to new file
    gr = save_f.create_group('init')
    for key in init_group.keys():
        # save initialization data
        gr.create_dataset(key, data=np.array(init_group.get(key)))
    # save calculated values ( mean, var, mf, rmf, mf / rmf times, mean / var times)
    save_f.create_dataset('sum_lin_vars', data=_lin_sum)
    save_f.create_dataset('sum_squared_vars', data=_square_sum)
    save_f.create_dataset('sum_squared_hit_rate_caches', data=_square_sum_hit_rate_caches)
    save_f.create_dataset("sim_times", data=_times)
    save_f.create_dataset("nr_simulations", data=_nr_simulations)
    # change to mf_values, rmf_values
    save_f.create_dataset("mf_values", data=mf_values, compression="gzip", compression_opts=9)
    save_f.create_dataset("rmf_values", data=mf_values + rmf_correction, compression="gzip", compression_opts=9)
    save_f.create_dataset("mf_times", data=rmf_times, compression="gzip", compression_opts=9)

    load_f.close()
    save_f.close()
    if output:
        print("Values successfully saved.")

def safe_mean_std_hit_rates(file_path, output=False):
    from tools.analysis_reproduction import save_load
    if output:
        print("Saving hit rate mean, std and (refined) mean field approx for\n\t" + file_path)
    from simulationFunctions.simulationTools import calculate_lin_square_sum

    # load file
    load_f = h5py.File(file_path, 'r')
    init_group = load_f.get("init")

    # get initial parameters
    nr_items = int(np.array(init_group.get('nr_items')))
    zipf_alpha = float(np.array(init_group.get('zipf_alpha')))
    simulation_time = int(np.array(init_group.get('simulation_time')))
    cache_sizes = np.array(init_group.get('cache_sizes'))
    initial_state = np.array(init_group.get('initial_state'))

    # calculate values
    if output:
        print("Calculating lin and squared terms")
    # _lin_sum, _square_sum, _square_sum_hit_rate_caches, _times, _nr_simulations = calculate_lin_square_sum(load_f, time_steps=400)
    # _sim_mean_values, _times, _nr_items = save_load(calculate_sim_mean)(load_f, normalize=False)
    # _sim_var_values, _, _ = save_load(calculate_sim_var)(load_f, normalize=False)

    model = initialize_model(nr_items=nr_items, zipf_alpha=zipf_alpha,
                             cache_sizes=cache_sizes, initial_state=initial_state)

    if output:
        print("Calculating MF and RMF")
    rmf_times, mf_values, rmf_correction, _ = model.meanFieldExpansionTransient(order=1, time=simulation_time)

    # set filename
    cache_sizes_sting = ""
    for cache in cache_sizes:
        cache_sizes_sting += str(cache) + "-"
    cache_sizes_sting = cache_sizes_sting[:-1]

    name = "sim_mean_var__nr_items-" + str(nr_items) + "_zipf_alpha-" + str(zipf_alpha) \
           + "_simulation_time-" + str(simulation_time) + "_cache_sizes-" + cache_sizes_sting

    save_f = h5py.File('cache/' + name + '.h5', 'w')

    # save values to new file
    gr = save_f.create_group('init')
    for key in init_group.keys():
        # save initialization data
        gr.create_dataset(key, data=np.array(init_group.get(key)))
    # save calculated values ( mean, var, mf, rmf, mf / rmf times, mean / var times)
    save_f.create_dataset('sum_lin_vars', data=_lin_sum)
    save_f.create_dataset('sum_squared_vars', data=_square_sum)
    save_f.create_dataset('sum_squared_hit_rate_caches', data=_square_sum_hit_rate_caches)
    save_f.create_dataset("sim_times", data=_times)
    save_f.create_dataset("nr_simulations", data=_nr_simulations)
    # change to mf_values, rmf_values
    save_f.create_dataset("mf_values", data=mf_values, compression="gzip", compression_opts=9)
    save_f.create_dataset("rmf_values", data=mf_values + rmf_correction, compression="gzip", compression_opts=9)
    save_f.create_dataset("mf_times", data=rmf_times, compression="gzip", compression_opts=9)

    load_f.close()
    save_f.close()
    if output:
        print("Values successfully saved.")

def update_mean_var():
    # TODO implement
    pass

def load_mean_var_mf(file_path, output=False):
    """
    Loads lin, squared sum of simulations as well as the mean field and refined mean field approximation

    :param file_path:
    :param output:
    :return:
    """
    # read file
    f = h5py.File(file_path, 'r')
    # initialize dictionary
    value_dict = dict()
    if output:
        print("\nLoaded " + str(file_path))
        print("\nModel initialization parameters: \n")
    init_group = f.get("init")
    # print(f.keys())
    for key in init_group.keys():
        if output:
            print(str(key) + " :")
            print(np.array(init_group.get(key)))
            print("--------------------")
        value_dict[str(key)] = np.array(init_group.get(key))
    for key in f.keys():
        if key == 'init':
            continue
        value_dict[str(key)] = np.array(f.get(key))
    f.close()
    return value_dict

if __name__ == "__main__":
    import os
    from simulationFunctions.master_load_function import gather_simulations
    from simulationFunctions.simulationTools import calculate_hit_rates

    load_values = False
    if not load_values:
        # calculate values
        cache_path = os.path.join(os.getcwd(), "../notebooks/Simulations/cache")
        filenames = gather_simulations(nr_items=10, zipf_alpha=0.8, cache_sizes=[4], cache_path=cache_path)
        for file in filenames:
            path = os.path.join(os.getcwd(), "../notebooks/Simulations/cache", file)
            safe_mean_var_mf(file_path=path, output=True)
    else:
        # load values
        cache_path = os.path.join(os.getcwd(), "cache")
        filenames = gather_simulations(nr_items=35, zipf_alpha=0.8, cache_sizes=[5, 5], cache_path=cache_path)
        for file in filenames:
            path = os.path.join(os.getcwd(), "cache", file)
            value_dict = load_mean_var_mf(file_path=path, output=True)
            calculate_hit_rates(value_dict)
