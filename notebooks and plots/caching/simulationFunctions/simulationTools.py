from tools.analysis_reproduction import save_load

import numpy as np

from scipy.interpolate import interp1d
from cacheModel import initialize_model, zipf

def calculate_sim_mean(hdf5_file, time_steps=200, normalize=True):
    """
    Calculate the average values ( expectation ) of a set of simulationFunctions saved in a hdf5_file
    :param hdf5_file - opened hdf5 file
    :param time_steps - at how many equidistant time steps the interpolation should be done
    :param normalize - if results should be divided by the nr simulationFunctions used for calulation ( of simulationFunctions are
                        saved in different files this should be false )
    :return: np array ( calculated expectation ), [nr. simulations if not normalized], np array ( interpolation times )
    """
    init_gr = hdf5_file.get("init")
    simulation_time = np.array(init_gr.get('simulation_time'))
    nr_simulations = np.array(init_gr.get("nr_simulations"))
    caches = np.array(init_gr.get("cache_sizes")).size + 1
    # initialize the mean values (simulation data come as array nr_items*nr_caches)
    sim_mean = np.zeros(shape=(np.array(init_gr.get("nr_items")) * caches, time_steps))
    # print("Calculating mean for " + str(nr_simulations) + " simulations.")
    interpolation_times = np.linspace(0, simulation_time, time_steps)
    for i in range(int(nr_simulations)):
        sim_gr = (hdf5_file.get("sim_" + str(i)))
        times = np.array(sim_gr.get("times"))
        data = np.array(sim_gr.get("data"))
        # append new interpolation to list
        # important that result from interpolations need to be transposed again to obtain (nr_item, state) sized array
        interpolation = interp1d(times, data, axis=0)
        # interpolation = interp1d(times, data.transpose())
        for k, time in enumerate(interpolation_times):
            interpolations_values = interpolation(time)
            sim_mean[:, k] += interpolation(time)  # .transpose()
    if normalize:
        sim_mean *= 1 / float(nr_simulations)
        return sim_mean, interpolation_times
    else:
        return sim_mean, interpolation_times, nr_simulations

def calculate_sim_mean_multi(hdf5_files):
    sim_mean = 0
    normalization_factor = 0
    for file in hdf5_files:
        _sim_mean, interpolation_times, _normalization_factor = calculate_sim_mean(file, normalize=False)
        sim_mean += _sim_mean
        normalization_factor += _normalization_factor
    sim_mean *= 1 / float(normalization_factor)
    return sim_mean, interpolation_times


def calculate_sim_var(hdf5_file, time_steps=200):
    """
    Rework
    :param hdf5_file:
    :param time_steps: !time steps need to be the same as in mean calculation
    :param normalize:
    :return:
    """
    init_gr = hdf5_file.get("init")
    _sim_mean, _ = save_load(calculate_sim_mean)(hdf5_file)
    simulation_time = np.array(init_gr.get('simulation_time'))
    nr_simulations = np.array(init_gr.get("nr_simulations"))
    caches = np.array(init_gr.get("cache_sizes")).size + 1
    _sim_var = np.zeros(shape=(np.array(init_gr.get("nr_items")) * caches, time_steps))
    interpolation_times = np.linspace(0, simulation_time, 200)
    # for file in hdf5_files:
    #     init_gr = file.get("init")
    #     nr_simulations += np.array(init_gr.get("nr_simulations"))
    for i in range(int(nr_simulations)):
        sim_gr = (hdf5_file.get("sim_" + str(i)))
        times = np.array(sim_gr.get("times"))
        data = np.array(sim_gr.get("data"))
        # no transpose needed anymore
        interpolation = interp1d(times, data, axis=0)
        # old (sir.simulate output changes and thus how it is saved)
        # interpolation = interp1d(times, data.transpose())
        for k, time in enumerate(interpolation_times):
            _sim_var[:, k] += np.square(interpolation(time) - _sim_mean[:, k])
    # return sim_var, interpolation_times
    _sim_var *= 1 / float(nr_simulations)
    return _sim_var, interpolation_times


def calculate_lin_square_sum(hdf5_file, time_steps=200):
    # get important initialization parameters
    init_gr = hdf5_file.get("init")
    _simulation_time = np.array(init_gr.get('simulation_time'))
    _nr_simulations = np.array(init_gr.get("nr_simulations"))
    _cache_sizes = np.array(init_gr.get("cache_sizes"))
    _caches = _cache_sizes.size + 1
    _nr_items = np.array(init_gr.get("nr_items"))

    # initialize arrays
    _square_sum = np.zeros(shape=(_nr_items * _caches, time_steps))
    _lin_sum = np.zeros(shape=(_nr_items * _caches, time_steps))
    _square_sum_hit_rate_caches = np.zeros(shape=(_caches, time_steps))
    # define interpolation times
    _interpolation_times = np.linspace(0, _simulation_time, time_steps)

    # initialize model for hit rates
    model = initialize_model(nr_items=_nr_items,zipf_alpha=np.array(init_gr.get("zipf_alpha")),
                             cache_sizes=_cache_sizes,initial_state=np.array(init_gr.get("initial_state")))
    # calculate terms
    for i in range(int(_nr_simulations)):
        sim_gr = (hdf5_file.get("sim_" + str(i)))
        times = np.array(sim_gr.get("times"))
        data = np.array(sim_gr.get("data"))
        interpolation = interp1d(times, data, axis=0)
        for k, time in enumerate(_interpolation_times):
            _square_sum[:, k] += np.square(interpolation(time))
            _lin_sum[:, k] += interpolation(time)
            for cache in range(_caches):
                _square_sum_hit_rate_caches[cache, k] += np.square(model.hit_rate(interpolation(time), cache))

    _square_sum_hit_rate_caches = np.transpose(_square_sum_hit_rate_caches)
    _lin_sum = np.transpose(_lin_sum)
    _square_sum = np.transpose(_square_sum)
    return _lin_sum, _square_sum, _square_sum_hit_rate_caches, _interpolation_times, _nr_simulations


def calculate_mean_var_std_from_sums(value_dict):
    """

    :param value_dict: expects dictionary from run_save_simulations.load_mean_var_mf function.
    :return:
    """
    _nr_simulations = np.array(value_dict['nr_simulations'])

    _times = value_dict['sim_times']
    _sim_mean = (1 / _nr_simulations) * value_dict['sum_lin_vars']
    _sim_var = (1 / _nr_simulations) * value_dict['sum_squared_vars'] - np.square((1 / _nr_simulations) *
                                                                                 value_dict['sum_lin_vars'])
    _sim_std = np.sqrt(_sim_var)
    return _sim_mean, _sim_var, _sim_std, _times

def calculate_hit_rate_mean_std(hdf5_file, time_steps=200):
    # get important initialization parameters
    init_gr = hdf5_file.get("init")
    _simulation_time = np.array(init_gr.get('simulation_time'))
    _nr_simulations = np.array(init_gr.get("nr_simulations"))
    _cache_sizes = np.array(init_gr.get("cache_sizes"))
    _caches = _cache_sizes.size + 1
    _nr_items = np.array(init_gr.get("nr_items"))

    # initialize arrays
    _hit_mean_caches = np.zeros(shape=(_caches, time_steps))
    _hit_std_caches = np.zeros(shape=(_caches, time_steps))
    # define interpolation times
    _interpolation_times = np.linspace(0, _simulation_time, time_steps)

    # initialize model for hit rates
    model = initialize_model(nr_items=_nr_items,zipf_alpha=np.array(init_gr.get("zipf_alpha")),
                             cache_sizes=_cache_sizes,initial_state=np.array(init_gr.get("initial_state")))
    # calculate terms
    for i in range(int(_nr_simulations)):
        sim_gr = (hdf5_file.get("sim_" + str(i)))
        times = np.array(sim_gr.get("times"))
        data = np.array(sim_gr.get("data"))
        interpolation = interp1d(times, data, axis=0)
        for k, time in enumerate(_interpolation_times):
            for cache in range(_caches):
                _hit_mean_caches[cache, k] += model.hit_rate(interpolation(time), cache) / _nr_simulations

    for i in range(int(_nr_simulations)):
        sim_gr = (hdf5_file.get("sim_" + str(i)))
        times = np.array(sim_gr.get("times"))
        data = np.array(sim_gr.get("data"))
        interpolation = interp1d(times, data, axis=0)
        for k, time in enumerate(_interpolation_times):
            for cache in range(_caches):
                _hit_std_caches[cache, k] += \
                    np.power(model.hit_rate(interpolation(time), cache) - _hit_mean_caches[cache, k], 2) / _nr_simulations

    _hit_std_caches = np.sqrt(_hit_std_caches)

    _hit_mean_caches = np.transpose(_hit_mean_caches)
    _hit_std_caches = np.transpose(_hit_std_caches)

    return _hit_mean_caches, _hit_std_caches, _interpolation_times, _nr_simulations


def calculate_hit_rates(value_dict):
    # get values
    _sim_mean, _sim_var, _sim_std, _times = calculate_mean_var_std_from_sums(value_dict=value_dict)
    _mf_values = value_dict['mf_values']
    _rmf_values = value_dict['rmf_values']
    _mf_times = value_dict['mf_times']
    _square_sum_hit_rate_caches = (1 / value_dict['nr_simulations']) * value_dict['sum_squared_hit_rate_caches']

    nr_caches = value_dict["cache_sizes"].size + 1
    hit_rates_caches_mean = {}
    hit_rates_cache_std = {}
    hit_rates_cache_mf = {}
    hit_rates_cache_rmf = {}

    model = initialize_model(nr_items=value_dict['nr_items'], zipf_alpha=value_dict['zipf_alpha'],
                             cache_sizes=value_dict['cache_sizes'], initial_state=value_dict['initial_state'])

    for cache in range(nr_caches):
        hit_rates_caches_mean[cache] = []
        hit_rates_cache_std[cache] = []
        hit_rates_cache_mf[cache] = []
        hit_rates_cache_rmf[cache] = []

        for i, _ in enumerate(_times):
            mean_value = model.hit_rate(_sim_mean[i, :], cache)
            hit_rates_caches_mean[cache].append(mean_value)
            hit_rate_std = _square_sum_hit_rate_caches[i, cache] - np.square(mean_value)
            hit_rates_cache_std[cache].append(hit_rate_std)
            # var_value = model.hit_rate(_sim_std[i, :], cache)
            # hit_rates_cache_std[cache].append(var_value)
        for i, _ in enumerate(_mf_times):
            # format of mf is transposed!
            mf_value = model.hit_rate(_mf_values[i, :], cache)
            hit_rates_cache_mf[cache].append(mf_value)
            rmf_value = model.hit_rate(_rmf_values[i, :], cache)
            hit_rates_cache_rmf[cache].append(rmf_value)

        hit_rates_caches_mean[cache] = np.array(hit_rates_caches_mean[cache])
        hit_rates_cache_std[cache] = np.array(hit_rates_cache_std[cache])
        hit_rates_cache_mf[cache] = np.array(hit_rates_cache_mf[cache])
        hit_rates_cache_rmf[cache] = np.array(hit_rates_cache_rmf[cache])

    return {'hit_rate_sim_mean': hit_rates_caches_mean, 'hit_rate_sim_std': hit_rates_cache_std,
            'hit_rate_mf': hit_rates_cache_mf, 'hit_rate_rmf': hit_rates_cache_rmf}

    # hit_rates_confidence = {}
    # for cache in range(nr_caches):
    #     hit_rates_confidence[cache] = 2 * hit_rates_cache_std[cache] / np.sqrt(number_simulations)


def calculate_std(hdf5_files, time_steps=200):
    if not isinstance(hdf5_files, list):
        # check if input is only one file
        hdf5_files = [hdf5_files]
    file = hdf5_files[0]
    _sim_var, _interpolation_times = save_load(calculate_sim_var)(file, time_steps)
    return np.sqrt(_sim_var), _interpolation_times


def cal_total_error(mean_field, mean_field_times, sim_mean, sim_mean_times, time, time_steps=200):
    print("Calculating total error of the approximation and simulated expectation.")

    _total_error = np.zeros(shape=time_steps)
    interpolation_times = np.linspace(0, time, time_steps)
    _interpolation = interp1d(np.array(mean_field_times), mean_field)
    for k, _time in enumerate(interpolation_times):
        _values = np.divide(sim_mean[:, :, k] - _interpolation(_time), sim_mean[:, :, k],
                            where=sim_mean[:, :, k] > 1e-6)
        _values = np.sum(_values)
        _total_error[k] = np.abs(_values)
        print("Total Error at time " + str(_time) + "\t" + str(_total_error[k]) + "; Summed error " + str(
            np.sum(_total_error)))
    return _total_error


def per_state_error(mean_field, mean_field_times, sim_mean, time, nr_items, time_steps=200):
    print("Calculating per state error of the approximation and simulated expectation.")

    _per_state_error = np.zeros(shape=time_steps)
    interpolation_times = np.linspace(0, time, time_steps)
    _interpolation = interp1d(np.array(mean_field_times), mean_field)
    for k, _time in enumerate(interpolation_times):
        # _values = np.divide(sim_mean[:, :, k] - _interpolation(_time), sim_mean[:, :, k],
        #                     where=sim_mean[:, :, k] > 1e-6)
        _values = sim_mean[:, :, k] - _interpolation(_time)
        _values = np.abs(_values)
        _per_state_error[k] = np.sum(_values) / float(nr_items)
        # print("Total Error at time " + str(_time) + "  \t" + str(_per_state_error[k]) + "; \t Summed error " + str(
        #     np.sum(_per_state_error)))
    return _per_state_error


def confidence_95(sim_mean, sim_var, nr_sim):
    """

    :param sim_mean:
    :param sim_var:
    :param nr_sim:
    :return:
    """
    bound_low = sim_mean - (2 / np.sqrt(float(nr_sim))) * np.sqrt(sim_var)
    bound_high = sim_mean + (2 / np.sqrt(float(nr_sim))) * np.sqrt(sim_var)
    return bound_low, bound_high


def error_simulation(sim_var, nr_sim):
    results = (1 / nr_sim) * (2 / np.sqrt(float(nr_sim))) * np.sqrt(sim_var)
    return np.sum(np.sum(results, axis=1), axis=0)


if __name__ == "__main__":
    from simulationFunctions.save_load_simulations import load_simulations, load_mean_var_mf, safe_mean_var_mf
    import sys, os

    current_dir = os.path.dirname(os.path.abspath("__file__"))
    # print(current_dir)
    file_path = os.path.join(current_dir, '..\\notebooks\\Simulations\\cache\\nr_simulations-2000_nr_items-5_alpha-0.8_simulation_time-100.0_2021-06-11_12-24-22.h5')

    open_file, _ = load_simulations(file_path=file_path)
    # value_dict = load_mean_var_mf(file_path=file_path, output=True)
    # calculate_lin_square_sum(open_file)
    safe_mean_var_mf(file_path=file_path)
    # safe_mean_var_mf(file_path="cache\\nr_simulations-2000_nr_items-20_alpha-0.8_simulation_time-200.0_2020-11-18_15-57-59.h5")
    # safe_mean_var_mf(file_path="cache\\nr_simulations-2000_nr_items-20_alpha-0.8_simulation_time-200.0_2020-11-18_16-01-54.h5")

    # calculate_hit_rates(value_dict=value_dict)
    # calculate_hit_rate_mean_std(open_file)
