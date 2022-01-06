### general tools for analysis and reproduction of results

# import
# packages
import numpy as np
# hdf5 to save tensors / data structures (i.e. data and times)
import h5py

# python tools
from time import time
from functools import wraps

save_load_functions = True
print_save_load_messages = True
time_functions = True


def save_load(func, stop_name_at_argument=None):
    """
    use hdf5 files
    """
    global save_load_functions
    @wraps(func)
    def _save_function(*args, **kwargs):
        name = None
        if hasattr(func, '__name__'):
            name = func.__name__
        elif hasattr(func, '__qualname__'):
            name = func.__qualname__
        else:
            raise Warning("Input has no name, thus can't be save and loaded.")
        for argument in args[:stop_name_at_argument]:
            name += "_"
            name += str(argument)
        for key in kwargs:
            kwargs[key]
            name += "_"
            name += str(key) + "-" + str(kwargs[key])
        # removing non valid symbols from name
        name = name.translate({ord(i): None for i in '<>/|"'})
        name = name.replace(' ', '_')
        # searching for results that have time and data dependencies
        time_dependence = False
        for _function_name in ['ode', 'simulate', 'meanFieldExpansionTransient']:
            if name.find(_function_name) != -1:
                time_dependence = True

        try:
            f = h5py.File('cache/' + name + '.h5', 'r')
            if time_dependence:
                # load time and data from file
                time = np.array(f['time'])
                data = np.array(f['data'])
                results = [time, data]
            else:
                # load data from file, always saved with dataset name 'data'
                results = np.array(f['data'])
            f.close()
            # with open("cache/" + name + ".npy", 'rb') as f:
            #     results = np.load(f)
            if print_save_load_messages:
                print("Function " + name + " has been loaded.")
        except:
            try:
                f.close()
            except:
                pass
            if time_dependence:
                results = func(*args, **kwargs)
                f = h5py.File('cache/' + name + '.h5', 'w')
                f.create_dataset('time', data=results[0], compression="gzip")
                f.create_dataset('data', data=results[1], compression="gzip")
            else:
                results = func(*args, **kwargs)
                f = h5py.File('cache/' + name + '.h5', 'w')
                f.create_dataset('data', data=results, compression="gzip")
            f.close()
            # old
            # with open("cache/" + name + ".npy", 'wb') as f:
            #     np.save(f, results)
            if print_save_load_messages:
                print("Function has been saved successfully at cache/" + name + " .")
        return results

    if save_load_functions is not None:
        if save_load_functions:
            return _save_function
        else:
            return func

def timer(func):
    """
    Timer decorator, does what the name suggests.

    Important -- To time a function, i.e. func(a, b), it has to be called like timer(func)(a, b). Timer wraps the
    function but NOT the arguments.
    :param func: function to wrap
    :return: wrapped callable function
    """
    global time_functions
    @wraps(func)
    def _measure_time(*args, **kwargs):
        if hasattr(func, '__name__'):
            print(f"Started to time '{func.__name__}'...")
        elif hasattr(func, '__qualname__'):
            print(f"Started to time '{func.__qualname__}' ...")
        else:
            print(f"Started to time...")
        start = time()
        try:
            return func(*args, **kwargs)
        finally:
            end = time()
            # get function name if existent and print time to console
            if hasattr(func, '__name__'):
                print(f"Execution time of '{func.__name__}': {(end - start) if end > start else 0} s")
            elif hasattr(func, '__qualname__'):
                print(f"Execution time of '{func.__qualname__}': {(end - start) if end > start else 0} s")
            else:
                print(f"Execution time: {(end - start) if end > start else 0} s")

    if time_functions is not None:
        if time_functions:
            return _measure_time
        else:
            return func


if __name__ == "__main__":
    save_load_functions = True
