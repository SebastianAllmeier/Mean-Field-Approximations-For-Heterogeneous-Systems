
import matplotlib.pyplot as plt
import numpy as np



class Plotting:
    """
    Class inheriting all plotting functions.
    """

    @staticmethod
    def generate_lines(data):
        # assumption: data has dimensions (nr_items, nr_states, times)
        lines = np.sum(data, axis=0) / data.shape[0]
        return lines

    @staticmethod
    def percentage_graph(times, system_states, labels=None, block=True):
        # Define name of states.


        # generate the lines to plot
        lines = Plotting.generate_lines(data=system_states)
        # generate the plot

        # plot lines
        if labels is None:
            labels = ["S", "I", "R"]

        plt.figure()
        for index, line in enumerate(lines):
            plt.plot(times, lines[index, :], label="{}".format(labels[index]), alpha=0.7)

        # Set graph propertie / layout
        plt.ylim(0, 1.1)
        plt.xlim(0, None)
        plt.legend()

        # Block program so graph can be viewed. Should be enabled at least for the last plot.
        if block:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.1)

    @staticmethod
    def multiple_percentage_graphs(*args, block=True):
        # takes multiple dictionaries with arguments :
        # data, times, labels mandatory
        # alpha optional (alpha channel for lines - in [0, 1.])
        fig = plt.figure()
        save = True

        # plotting the dictionaries
        for arg in args:
            lines = Plotting.generate_lines(data=arg["data"])
            if "alpha" in arg:
                alpha = arg["alpha"]
            else:
                alpha = 1.0
            for index, line in enumerate(lines):
                line, = plt.plot(arg["times"], lines[index, :], label="{}".format(arg["labels"][index]), alpha=alpha)
                if "colors" in arg:
                    line.set_color(arg["colors"][index])


        # Set graph propertie / layout
        plt.ylim(0, 1.1)
        plt.xlim(0, None)
        plt.legend()

        # Block program so graph can be viewed. Should be enabled at least for the last plot.
        if block:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.1)
        if save:
            fig.savefig("test_fig.pdf")

    @staticmethod
    def multiple_percentage_graphs_subplot(subplot, *args):
        # takes multiple dictionaries with arguments :
        # data, times, labels mandatory
        # alpha optional (alpha channel for lines - in [0, 1.])
        # plotting the dictionaries

        for arg in args:
            lines = Plotting.generate_lines(data=arg["data"])
            if "alpha" in arg:
                alpha = arg["alpha"]
            else:
                alpha = 1.0
            for index, line in enumerate(lines):
                line, = subplot.plot(arg["times"], lines[index, :], label="{}".format(arg["labels"][index]),
                                 alpha=alpha)
                if "colors" in arg:
                    line.set_color(arg["colors"][index])

        # Set graph propertie / layout
        plt.ylim(0, 1.1)
        plt.xlim(0, None)
        plt.legend()

    # @staticmethod
    # def percentage_graph_sim_mean_old(times, system_states, block=True):
    #     # Define name of states.
    #     states = ["S", "I", "R"]
    #     lines = Plotting.generate_sim_exp_lines(data=system_states)
    #     # Time_steps and system_states must have the same length.
    #     system_size = system_states.shape[0]
    #     lines = np.zeros(shape=(system_states.shape[1], len(times)))
    #     for step in range(len(times)):
    #         # Append the percentage of entities in the respective state to the corresponding entry in lines matrix.
    #         # Summing over all entities and dividing by amount of entities to get percentage in states.
    #         for state, value in enumerate(np.sum(system_states[:, :, step], axis=0) / system_size):
    #             lines[state, step] = value
    #     # Plot the lines.
    #     plt.figure()
    #     for index, line in enumerate(lines):
    #         plt.plot(times, lines[index, :], label="{}".format(states[index]), alpha=0.7)
    #
    #     # Set graph properties.
    #     plt.ylim(0, 1.1)
    #     plt.xlim(0, None)
    #     plt.legend()
    #     # Block program so graph can be viewed. Should be enabled at least for the last plot.
    #     if block:
    #         plt.show()
    #     else:
    #         plt.draw()
    #         plt.pause(0.1)
    #
    # @staticmethod
    # def ode_against_simulation_percentage(times_ctmc, states_ctmc, times_ode, states_ode, block=True):
    #     # Define name of states.
    #     states = ["S", "I", "R"]
    #     # Time_steps and system_states must have the same length.
    #     system_size_ctmc = states_ctmc[0].shape
    #     system_size_ode = states_ode[0].shape
    #
    #     # Plot ctmc lines.
    #     lines_ctmc = np.zeros(shape=(system_size_ctmc[1], len(times_ctmc)))
    #     for step in range(len(times_ctmc)):
    #         # Append the percentage of entities in the respective state to the corresponding entry in lines matrix.
    #         # Summing over all entities and dividing by amount of entities to get percentage in states.
    #         for state, value in enumerate(np.sum(states_ctmc[step], axis=0) / system_size_ctmc[0]):
    #             lines_ctmc[state][step] = value
    #     # Plot the lines.
    #     # fig = plt.figure() not used
    #     for index, line in enumerate(lines_ctmc):
    #         plt.plot(times_ctmc, lines_ctmc[index, :], label="{} (ctmc)".format(states[index]), alpha=0.7)
    #
    #     # Plot ode lines.
    #     lines_ode = np.zeros(shape=(system_size_ode[1], len(times_ode)))
    #     for step in range(len(times_ode)):
    #         # Append the percentage of entities in the respective state to the corresponding entry in lines matrix.
    #         # Summing over all entities and dividing by amount of entities to get percentage in states.
    #         for state, value in enumerate(np.sum(states_ode[step], axis=0) / system_size_ode[0]):
    #             lines_ode[state][step] = value
    #     # Plot the lines.
    #     # fig = plt.figure()
    #     for index, line in enumerate(lines_ode):
    #         plt.plot(times_ode, lines_ode[index, :], '--', label="{} (ode)".format(states[index]), alpha=0.3)
    #
    #     # Set graph properties.
    #     plt.ylim(0, 1.1)
    #     plt.xlim(0, None)
    #     plt.legend()
    #     # Block program so graph can be viewed. Should be enabled at least for the last plot.
    #     if block:
    #         plt.show()
    #     else:
    #         plt.draw()
    #         plt.pause(0.1)
    #
    # @staticmethod
    # def ode_vs_sim_vs_refine_percentage(times_ctmc, states_ctmc, times_ode, states_ode,
    #                                     times_refine, states_refine, block=True):
    #     # Define name of states.
    #     states = ["S", "I", "R"]
    #     # Time_steps and system_states must have the same length.
    #     system_size_ctmc = states_ctmc[0].shape
    #     system_size_ode = states_ode[0].shape
    #     system_size_refine = states_refine[0].shape
    #
    #     # Plot ctmc lines.
    #     lines_ctmc = np.zeros(shape=(system_size_ctmc[1], len(times_ctmc)))
    #     for step in range(len(times_ctmc)):
    #         # Append the percentage of entities in the respective state to the corresponding entry in lines matrix.
    #         # Summing over all entities and dividing by amount of entities to get percentage in states.
    #         for state, value in enumerate(np.sum(states_ctmc[step], axis=0) / system_size_ctmc[0]):
    #             lines_ctmc[state][step] = value
    #     # Plot the lines.
    #     # fig = plt.figure() - not used
    #     for index, line in enumerate(lines_ctmc):
    #         plt.plot(times_ctmc, lines_ctmc[index, :], label="{} (ctmc)".format(states[index]), alpha=0.9)
    #
    #     # Plot ode lines.
    #     lines_ode = np.zeros(shape=(system_size_ode[1], len(times_ode)))
    #     for step in range(len(times_ode)):
    #         # Append the percentage of entities in the respective state to the corresponding entry in lines matrix.
    #         # Summing over all entities and dividing by amount of entities to get percentage in states.
    #         for state, value in enumerate(np.sum(states_ode[step], axis=0) / system_size_ode[0]):
    #             lines_ode[state][step] = value
    #     # Plot the lines.
    #     # fig = plt.figure()
    #     for index, line in enumerate(lines_ode):
    #         plt.plot(times_ode, lines_ode[index, :], '--', label="{} (ode)".format(states[index]), alpha=0.6)
    #
    #     # Plot refinement lines.
    #     lines_refine = np.zeros(shape=(system_size_refine[1], len(times_refine)))
    #     for step in range(len(times_refine)):
    #         # Append the percentage of entities in the respective state to the corresponding entry in lines matrix.
    #         # Summing over all entities and dividing by amount of entities to get percentage in states.
    #         for state, value in enumerate(np.sum(states_refine[step], axis=0) / system_size_refine[0]):
    #             lines_refine[state][step] = value
    #     # Plot the lines.
    #     # fig = plt.figure()
    #     for index, line in enumerate(lines_ode):
    #         plt.plot(times_refine, lines_refine[index, :], '--', label="{} (refined)".format(states[index]), alpha=0.3)
    #
    #     # Set graph properties.
    #     plt.ylim(0, 1.1)
    #     plt.xlim(0, None)
    #     plt.legend()
    #     # Block program so graph can be viewed. Should be enabled at least for the last plot.
    #     if block:
    #         plt.show()
    #     else:
    #         plt.draw()
    #         plt.pause(0.1)
