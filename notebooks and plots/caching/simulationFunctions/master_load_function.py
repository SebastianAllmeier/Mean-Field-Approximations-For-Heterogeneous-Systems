import sys

sys.path.append('../')

import os
from simulationFunctions.save_load_simulations import load_simulations

import pandas as pd

import numpy as np

## import functions for batch simulation
from simulationFunctions.save_load_simulations import load_simulations
from simulationFunctions.simulationTools import *

## import caching wrapper
from tools.analysis_reproduction import save_load
import tools.analysis_reproduction

import cacheModel


def gather_simulations(nr_items=None, zipf_alpha=None, cache_sizes=None, output=False, cache_path=None):
    dataFrame1 = pd.DataFrame(columns=['nr_items', 'zipf_alpha', 'cache_sizes', 'simulation_time', 'nr_simulations',
                                       'initial_state', 'filename'])
    file_nr = 0
    if cache_path is None:
        for filename in os.listdir(os.path.join(os.getcwd(), "../notebooks/Simulations/cache")):
            try:
                a, sim_information = load_simulations("cache/" + filename, output=False)
            except (OSError, AttributeError):
                continue
            a.close()
            sim_information['filename'] = filename
            if file_nr == 0:
                dataFrame1
            dataFrame1.loc[file_nr] = sim_information
            file_nr += 1
    else:
        for filename in os.listdir(cache_path):
            try:
                a, sim_information = load_simulations(cache_path + "/" + filename, output=False)
            except (OSError, AttributeError):
                continue
            a.close()
            sim_information['filename'] = filename
            if file_nr == 0:
                dataFrame1
            dataFrame1.loc[file_nr] = sim_information
            file_nr += 1
    if output:
        print(dataFrame1.iloc[:, 0:-2])
    # print(dataFrame1)
    # request_indices = dataFrame1.nr_items[dataFrame1.nr_items == 20].index.tolist()
    # print(request_indices)
    # for index in request_indices:
    #    print(dataFrame1.filename[index])

    search_dict = {'zipf_alpha': zipf_alpha, 'nr_items': nr_items}
    search_dict['cache_sizes'] = np.array(cache_sizes) if cache_sizes is not None else None
    sub_frame = dataFrame1
    for _key in search_dict:
        if search_dict[_key] is not None:
            if _key == 'cache_sizes':
                index_list = []
                for _i, cache_specs in enumerate(sub_frame[_key]):
                    try:
                        if (cache_specs == search_dict[_key]).all():
                            index_list.append(True)
                        else:
                            index_list.append(False)
                    except:
                        index_list.append(False)
                sub_frame = sub_frame.loc[index_list]
            else:
                sub_frame = sub_frame.loc[dataFrame1[_key] == search_dict[_key]]
    filenames = sub_frame.filename.to_numpy().tolist()
    return filenames


# def wrapper_function(nr_items=20, zipf_alpha=0.8, cache_sizes=[5, 3], cache_path=None, calculate_rmf=False):
#     # load files
#     filenames = gather_simulations(nr_items=nr_items, zipf_alpha=zipf_alpha, cache_sizes=cache_sizes,
#                                    cache_path=cache_path)
#
#     # open files
#     file_path = os.path.join("Simulations", "../notebooks/Simulations/cache", filenames[0])
#     open_h5_file, initial_values = load_simulations(file_path)
#
#     # calculate mean, var, std
#     sim_mean, sim_mean_times = save_load(calculate_sim_mean)(open_h5_file)
#     sim_var, sim_var_times = save_load(calculate_sim_var)(open_h5_file)
#     sim_std, sim_std_times = save_load(calculate_std)(open_h5_file)
#
#     # initialize the model
#     zipf_distribution = cacheModel.zipf(initial_values["nr_items"],
#                                         initial_values["zipf_alpha"])
#
#     model = cacheModel.cacheRANDmDDPP(zipf_distribution, initial_values["cache_sizes"])
#     model.set_initial_state(initial_values["initial_state"])
#
#     # calculate mean field approx.
#     # watch out for cached ode results, model difference is not registered when loading
#
#     mf_times, mf_values = model.ode(time=initial_values['simulation_time'])
#     if calculate_rmf:
#         rmf_times, rmf_values, _, _ = model.meanFieldExpansionTransient(order=1, time=initial_values['simulation_time'])
#
#     # put into simulationTools
#     nr_caches = initial_values["cache_sizes"].size + 1
#     hit_rates_caches = {}
#     hit_rates_cache_std = {}
#     hit_rates_cache_mf = {}
#     hit_rates_cache_rmf = {}
#
#     for cache in range(nr_caches):
#         hit_rates_caches[cache] = []
#         hit_rates_cache_std[cache] = []
#         hit_rates_cache_mf[cache] = []
#         hit_rates_cache_rmf[cache] = []
#
#         for i, _ in enumerate(sim_mean_times):
#             mean_value = model.hit_rate(sim_mean[:, i], cache)
#             hit_rates_caches[cache].append(mean_value)
#             var_value = model.hit_rate(sim_std[:, i], cache)
#             hit_rates_cache_std[cache].append(var_value)
#         for i, _ in enumerate(mf_times):
#             # format of mf is transposed!
#             mf_value = model.hit_rate(mf_values[i, :], cache)
#             hit_rates_cache_mf[cache].append(mf_value)
#         if calculate_rmf:
#             for i, _ in enumerate(rmf_times):
#                 rmf_value = model.hit_rate(rmf_values[i, :], cache)
#                 hit_rates_cache_rmf[cache].append(rmf_value)
#         hit_rates_caches[cache] = np.array(hit_rates_caches[cache])
#         hit_rates_cache_std[cache] = np.array(hit_rates_cache_std[cache])
#         hit_rates_cache_mf[cache] = np.array(hit_rates_cache_mf[cache])
#         if calculate_rmf:
#             hit_rates_cache_rmf[cache] = np.array(hit_rates_cache_rmf[cache])
#
#     number_simulations = initial_values["nr_simulations"]
#
#     hit_rates_confidence = {}
#     for cache in range(nr_caches):
#         hit_rates_confidence[cache] = 2 * hit_rates_cache_std[cache] / np.sqrt(number_simulations)
#
#     conf_95 = 2 * sim_std / np.sqrt(number_simulations - 1)
#
#     value_dict = {}
#     return value_dict




if __name__ == "__main__":
    gather_simulations(nr_items=20, zipf_alpha=1.2, cache_sizes=None, output=True)
