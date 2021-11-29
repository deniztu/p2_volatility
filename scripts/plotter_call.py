# -*- coding: utf-8 -*-
"""
This script outputs plots

@author: Deniz
"""

import numpy as np
import pdb

from classes.plotter import plotter

my_plotter = plotter(path_to_data = 'data\\rnn_raw_data',
                     path_to_save_plot = 'plots\\')


'''
CALIBRATION PLOTS
'''

'''
sd .1
'''

my_plotter.calibration_plot(zip_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_{}_fixed_res_rt_bin_p_{}_a_4_n_300_run_{}.zip',
                            file_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_{}_fixed_res_rt_bin_p_{}_a_4_n_300_run_{}_rin_{}.csv',
                            num_runs = 3,
                            num_instances = 2,
                            test_sd_array = np.arange(0.02, 0.34, 0.02),
                            main_title='RNN trained on SD .1, tested on binary independant 4-armed gaussian restless bandits',
                            x_lab = 'Test SD',
                            y_lab = 'Accuracy',
                            num_rins=3)  

'''
sd .05
'''

my_plotter.calibration_plot(zip_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_05_a_4_n_300_te_50000_id_{}.zip',
                            file_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_05_a_4_n_300_te_50000_id_{}_res_rt_bin_p_{}_n_300_run_{}.csv',
                            num_runs = 1,
                            num_instances = 10,
                            test_sd_array = np.arange(0.02, 0.34, 0.02),
                            main_title='RNN trained on SD .05, tested on binary independant 4-armed gaussian restless bandits',
                            x_lab = 'Test SD',
                            y_lab = 'Accuracy')

'''
meta-volatility
'''

my_plotter.calibration_plot(zip_name = 'rnn_n_f_p_0_met_rt_bin_d_f_p_n_a_4_n_300_te_50000_id_{}.zip',
                            file_name = 'rnn_n_f_p_0_met_rt_bin_d_f_p_n_a_4_n_300_te_50000_id_{}_res_rt_bin_p_{}_n_300_run_{}.csv',
                            num_runs = 1,
                            num_instances = 10,
                            test_sd_array = np.arange(0.02, 0.34, 0.02),
                            main_title='RNN trained on SD range[0.02 - 0.2], tested on binary independant 4-armed gaussian restless bandits',
                            x_lab = 'Test SD',
                            y_lab = 'Accuracy')

'''
all rnns hue
'''

my_plotter.calibration_plot_hue(zip_name = ['rnn_n_f_p_0_met_rt_bin_d_f_p_n_a_4_n_300_te_50000_id_{}.zip', 
                                        'rnn_n_f_p_0_res_rt_bin_d_f_p_0_05_a_4_n_300_te_50000_id_{}.zip',
                                        'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_{}.zip'],
                            file_name_suffix = '_res_rt_bin_p_{}_n_300_run_{}.csv',
                            num_runs = 1,
                            num_instances = 10,
                            test_sd_array = np.arange(0.02, 0.34, 0.02),
                            main_title='All RNNs (SD = .05, SD = .1, SD range[0.02 - 0.2]), tested on binary independant 4-armed gaussian restless bandits',
                            x_lab = 'Test SD',
                            y_lab = 'Accuracy')

'''
N_SWITCHES BY TEST SD PLOT
'''


'''
sd .1
'''

my_plotter.switches_by_test_sd_plot(zip_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_{}.zip',
                            file_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_{}_res_rt_bin_p_{}_n_300_run_{}.csv',
                            num_runs = 1,
                            num_instances = 10,
                            test_sd_array = np.arange(0.02, 0.34, 0.02),
                            main_title='RNN trained on SD .1, tested on binary independant 4-armed gaussian restless bandits',
                            x_lab = 'Test SD',
                            y_lab = 'Number of switches') 

'''
sd .05
'''

my_plotter.switches_by_test_sd_plot(zip_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_05_a_4_n_300_te_50000_id_{}.zip',
                            file_name = 'rnn_n_f_p_0_res_rt_bin_d_f_p_0_05_a_4_n_300_te_50000_id_{}_res_rt_bin_p_{}_n_300_run_{}.csv',
                            num_runs = 1,
                            num_instances = 10,
                            test_sd_array = np.arange(0.02, 0.34, 0.02),
                            main_title='RNN trained on SD .05, tested on binary independant 4-armed gaussian restless bandits',
                            x_lab = 'Test SD',
                            y_lab = 'Number of switches')

'''
meta-volatility
'''

my_plotter.switches_by_test_sd_plot(zip_name = 'rnn_n_f_p_0_met_rt_bin_d_f_p_n_a_4_n_300_te_50000_id_{}.zip',
                            file_name = 'rnn_n_f_p_0_met_rt_bin_d_f_p_n_a_4_n_300_te_50000_id_{}_res_rt_bin_p_{}_n_300_run_{}.csv',
                            num_runs = 1,
                            num_instances = 10,
                            test_sd_array = np.arange(0.02, 0.34, 0.02),
                            main_title='RNN trained on SD range[0.02 - 0.2], tested on binary independant 4-armed gaussian restless bandits',
                            x_lab = 'Test SD',
                            y_lab = 'Number of switches')

'''
all rnns hue
'''

my_plotter.switches_by_test_sd_plot_hue(zip_name = ['rnn_n_f_p_0_met_rt_bin_d_f_p_n_a_4_n_300_te_50000_id_{}.zip', 
                                        'rnn_n_f_p_0_res_rt_bin_d_f_p_0_05_a_4_n_300_te_50000_id_{}.zip',
                                        'rnn_n_f_p_0_res_rt_bin_d_f_p_0_1_a_4_n_300_te_50000_id_{}.zip'],
                            file_name_suffix = '_res_rt_bin_p_{}_n_300_run_{}.csv',
                            num_runs = 1,
                            num_instances = 10,
                            test_sd_array = np.arange(0.02, 0.34, 0.02),
                            main_title='All RNNs (SD = .05, SD = .1, SD range[0.02 - 0.2]), tested on binary independant 4-armed gaussian restless bandits',
                            x_lab = 'Test SD',
                            y_lab = 'Number of switches')




