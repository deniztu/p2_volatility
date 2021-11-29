# -*- coding: utf-8 -*-
"""
Class for different plotting methods 

Created on Mon Sep 20 15:38:56 2021

@author: Deniz
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp
import seaborn as sns
from statsmodels.stats.multitest import multipletests # for multiple comparison corrections
import pdb

from scripts.helpers import zip2csv
from scripts.helpers import dot2_



class plotter():
    def __init__(self, path_to_data, path_to_save_plot):
        
        self.path_to_data = path_to_data
        self.path_to_save_plot = path_to_save_plot
    
        
    '''
    calibration plot
    
    boxplot with accuracy on y-axis and test_sd_array on x-axis
    
    note: test_sd_array must be a numpy.ndarray
    '''
    def calibration_plot(self, zip_name, file_name, 
                         num_runs, test_sd_array, num_instances, x_lab = '', 
                         y_lab = '', main_title = ''):
        
        # prepare a list of lists of length len(SDS_ARRAY), each list appends a run of a certain sd
        data_to_plot = [[] for _ in range(len(test_sd_array))]
        
        # x ticks are defined for the plot
        x_ticks = []
        for i in test_sd_array: x_ticks.append(dot2_(i))

        MY_MODELS = [zip_name]
        
        for m, zip_file in enumerate(MY_MODELS):
            
            for sd_i, sd in enumerate(test_sd_array):
                
                sd_ = dot2_(sd)
                
                for id_ in range(num_instances):
                    
                    # extract files
                    zip_handler = zip2csv(zip_file_name=zip_file.format(id_))
                            
                    for run in range(num_runs):
                        
                        # import pdb
                        # pdb.set_trace()
                        
                        file_name_i = file_name.format(id_, sd_, run)
                        zip_handler.extract_file(file_name_i)
                        
                        # uncomment if findlings code is used
                        # file_name = zip_file.replace('.zip', '_run_{}.csv').format(g)
                        df = pd.read_csv(file_name_i)
                        
                        # # omit first 25 choices    
                        # choice = df['choice'].iloc[25:len(df['choice'])]
                        
                        choice = df['choice']
                        
                        # calculate steady state accuracy TODO write a function
                        higher_rew = []
                        arms = int(zip_file.split('a')[1][1])
                        p_rews = [] 
                        for arm in range(arms): p_rews.append('p_rew_{}'.format(arm+1))
                        # tmp_arr = df[p_rews].iloc[25:len(df['choice'])]
                        tmp_arr = df[p_rews]
                        for row in range(tmp_arr.shape[0]): 
                            higher_rew.append(np.argmax(tmp_arr.iloc[row]))
                        
                        # uncomment if findlings code is used
                        # higher_rew = df['p_rew_1st_lever'] < 1- df['p_rew_1st_lever']
                        
                        ss_acc = sum(choice == higher_rew)/len(choice)
                        
                        # append data
                        data_to_plot[sd_i].append(ss_acc)
                            
                        # delete file
                        zip_handler.delete_file(file_name_i)
                    
        # prepare df for plotting
        plot_df = pd.DataFrame(np.transpose(data_to_plot)) # transpose for correct plotting
        # import pdb; pdb.set_trace()
        
        # run one-sample t-test against chance for each column in plot_df
        is_significant = ttest_1samp(plot_df, 1/arms, axis=0, nan_policy='raise')
        
        # correct for multiple comparisons
        corrected_pvals = multipletests(is_significant.pvalue, alpha = 1/arms, method = 'bonferroni')
        
        # print('corrected_pvals')
        # print(corrected_pvals)
        
        is_significant = ['*' if p_val < 0.05 else '' for p_val in corrected_pvals[1]] # corrected pvals return pvalues at second index
        
        # print('uncorrected_pvals')
        # print(is_significant.pvalue < 0.05)
                    
        # plot boxplot
        plt.figure(figsize=(8, 3))
        
        box = sns.boxplot(data=plot_df, 
                         palette="colorblind")
        
        box.axhline(1/arms, ls = '--', c = 'black')
        
        box.set(
            xlabel=x_lab, 
            ylabel=y_lab)
        
        box.set_title(main_title)
        box.set_xticklabels(x_ticks)            
        
         # create statistic annotations '*' if significant
        plt.plot([0], [1], lw=1.5, c='white')
        for i in range(len(test_sd_array)):
            plt.text(i, .95, is_significant[i], ha='center', va='bottom', color='black')
        
        plt.show()

    '''
    calibration plot shows all levels of rnn train next to each other
    
    boxplot with accuracy on y-axis and test_sd_array on x-axis
    
    note: test_sd_array must be a numpy.ndarray
    '''
    def calibration_plot_hue(self, zip_name, file_name_suffix, 
                         num_runs, test_sd_array, num_instances, x_lab = '', 
                         y_lab = '', main_title = ''):
        
        # prepare a list of lists of length len(SDS_ARRAY), each list appends a run of a certain sd
        data_to_plot = [[] for _ in range(len(test_sd_array))]
        
        # x ticks are defined for the plot
        x_ticks = []
        for i in test_sd_array: x_ticks.append(dot2_(i))

        MY_MODELS = zip_name
        
        for m, zip_file in enumerate(MY_MODELS):
            
            for sd_i, sd in enumerate(test_sd_array):
                
                sd_ = dot2_(sd)
                
                for id_ in range(num_instances):
                    
                    # extract files
                    zip_handler = zip2csv(zip_file_name=zip_file.format(id_))
                            
                    for run in range(num_runs):
                        
                        # import pdb
                        # pdb.set_trace()
                        
                        file_name_i = zip_file.format(id_).replace('.zip', file_name_suffix.format(sd_, run))
                        zip_handler.extract_file(file_name_i)
                        
                        # uncomment if findlings code is used
                        # file_name = zip_file.replace('.zip', '_run_{}.csv').format(g)
                        df = pd.read_csv(file_name_i)
                        
                        # # omit first 25 choices    
                        # choice = df['choice'].iloc[25:len(df['choice'])]
                        
                        choice = df['choice']
                        
                        # calculate steady state accuracy TODO write a function
                        higher_rew = []
                        arms = int(zip_file.split('a')[1][1])
                        p_rews = [] 
                        for arm in range(arms): p_rews.append('p_rew_{}'.format(arm+1))
                        # tmp_arr = df[p_rews].iloc[25:len(df['choice'])]
                        tmp_arr = df[p_rews]
                        for row in range(tmp_arr.shape[0]): 
                            higher_rew.append(np.argmax(tmp_arr.iloc[row]))
                        
                        # uncomment if findlings code is used
                        # higher_rew = df['p_rew_1st_lever'] < 1- df['p_rew_1st_lever']
                        
                        ss_acc = sum(choice == higher_rew)/len(choice)
                        
                        # append data
                        data_to_plot[sd_i].append(ss_acc)
                            
                        # delete file
                        zip_handler.delete_file(file_name_i)
                    
        # prepare df for plotting
        plot_df = pd.DataFrame(np.transpose(data_to_plot)) # transpose for correct plotting
        
        # import pdb; pdb.set_trace()
        
        # run one-sample t-test against chance for each column in plot_df
        is_significant = ttest_1samp(plot_df, 1/arms, axis=0, nan_policy='raise')
        
        # creater x and y for hue
        my_x = np.array([np.arange(0.02, 0.34, 0.02)]*10*3).flatten()
        my_y = np.array(plot_df).flatten()
        
        my_hue = ['.02 - .2']*10*16 + ['.05']*10*16 + ['.1']*10*16
        
        plot_df = pd.DataFrame({'test_sd':my_x, 'accuracy': my_y, 'train_sd': my_hue})
        
        # correct for multiple comparisons
        corrected_pvals = multipletests(is_significant.pvalue, alpha = 1/arms, method = 'bonferroni')
        
        print('corrected_pvals')
        print(corrected_pvals)
        
        print('uncorrected_pvals')
        print(is_significant.pvalue < 0.05)
        # create statistic annotations '*' if significant
        is_significant = ['*' if p_val < 0.05 else '' for p_val in is_significant.pvalue]
        
                    
        # plot boxplot
        plt.figure(figsize=(8, 3))
        
        box = sns.boxplot(x = 'test_sd', y = 'accuracy', hue = 'train_sd', data=plot_df, 
                         palette="colorblind")
        
        box.axhline(1/arms, ls = '--', c = 'black')
        
        box.set(
            xlabel=x_lab, 
            ylabel=y_lab)
        
        box.set_title(main_title)
        box.set_xticklabels(x_ticks)            
        
        plt.show()
        
    '''
    plot number of switches as a funtion of test_sd
    '''
        
    def switches_by_test_sd_plot(self, zip_name, file_name, 
                 num_runs, test_sd_array, num_instances, x_lab = '', 
                 y_lab = '', main_title = ''):
        
        # prepare a list of lists of length len(SDS_ARRAY), each list appends a run of a certain sd
        data_to_plot = [[] for _ in range(len(test_sd_array))]
        
        # x ticks are defined for the plot
        x_ticks = []
        for i in test_sd_array: x_ticks.append(dot2_(i))

        MY_MODELS = [zip_name]
        
        for m, zip_file in enumerate(MY_MODELS):
            
            for sd_i, sd in enumerate(test_sd_array):
                
                sd_ = dot2_(sd)
                
                for id_ in range(num_instances):
                    
                    # extract files
                    zip_handler = zip2csv(zip_file_name=zip_file.format(id_))
                            
                    for run in range(num_runs):
                        
                        file_name_i = file_name.format(id_, sd_, run)
                        zip_handler.extract_file(file_name_i)
                        
                        # uncomment if findlings code is used
                        # file_name = zip_file.replace('.zip', '_run_{}.csv').format(g)
                        df = pd.read_csv(file_name_i)
                        
                        # # omit first 25 choices    
                        # choice = df['choice'].iloc[25:len(df['choice'])]
                        
                        choice = df['choice']
                        
                        # import pdb
                        # pdb.set_trace()
                        
                        # calculate number of switches
                        n_switches = 0
                        
                        for t, ch in enumerate(choice):
                            if t > 0:
                                if ch != choice[t-1]:
                                    n_switches += 1
                        
                        # append data
                        data_to_plot[sd_i].append(n_switches)
                            
                        # delete file
                        zip_handler.delete_file(file_name_i)
                    
        # prepare df for plotting
        plot_df = pd.DataFrame(np.transpose(data_to_plot)) # transpose for correct plotting
                                            
        # plot boxplot
        plt.figure(figsize=(8, 3))
        
        box = sns.boxplot(data=plot_df, 
                         palette="colorblind")
                
        box.set(
            xlabel=x_lab, 
            ylabel=y_lab)
        
        box.set_title(main_title)
        box.set_xticklabels(x_ticks)            
        
        plt.show()
        
    '''
    plot number of switches as a funtion of test_sd all levels of rnn train next to each other
    '''
        
    def switches_by_test_sd_plot_hue(self, zip_name, file_name_suffix, 
                         num_runs, test_sd_array, num_instances, x_lab = '', 
                         y_lab = '', main_title = ''):
        
        # prepare a list of lists of length len(SDS_ARRAY), each list appends a run of a certain sd
        data_to_plot = [[] for _ in range(len(test_sd_array))]
        
        # x ticks are defined for the plot
        x_ticks = []
        for i in test_sd_array: x_ticks.append(dot2_(i))

        MY_MODELS = zip_name
        
        for m, zip_file in enumerate(MY_MODELS):
            
            for sd_i, sd in enumerate(test_sd_array):
                
                sd_ = dot2_(sd)
                
                for id_ in range(num_instances):
                    
                    # extract files
                    zip_handler = zip2csv(zip_file_name=zip_file.format(id_))
                            
                    for run in range(num_runs):
                        
                        # import pdb
                        # pdb.set_trace()
                        
                        file_name_i = zip_file.format(id_).replace('.zip', file_name_suffix.format(sd_, run))
                        zip_handler.extract_file(file_name_i)
                        
                        # uncomment if findlings code is used
                        # file_name = zip_file.replace('.zip', '_run_{}.csv').format(g)
                        df = pd.read_csv(file_name_i)
                        
                        # # omit first 25 choices    
                        # choice = df['choice'].iloc[25:len(df['choice'])]
                        
                        choice = df['choice']
                        
                        # calculate number of switches
                        n_switches = 0
                        
                        for t, ch in enumerate(choice):
                            if t > 0:
                                if ch != choice[t-1]:
                                    n_switches += 1
                        
                        # append data
                        data_to_plot[sd_i].append(n_switches)
                            
                        # delete file
                        zip_handler.delete_file(file_name_i)
                    
        # prepare df for plotting
        plot_df = pd.DataFrame(np.transpose(data_to_plot)) # transpose for correct plotting
        
        # import pdb; pdb.set_trace()
        
        # creater x and y for hue
        my_x = np.array([np.arange(0.02, 0.34, 0.02)]*10*3).flatten()
        my_y = np.array(plot_df).flatten()
        
        my_hue = ['.02 - .2']*10*16 + ['.05']*10*16 + ['.1']*10*16
        
        plot_df = pd.DataFrame({'test_sd':my_x, 'accuracy': my_y, 'train_sd': my_hue})
        
        # plot boxplot
        plt.figure(figsize=(8, 3))
        
        box = sns.boxplot(x = 'test_sd', y = 'accuracy', hue = 'train_sd', data=plot_df, 
                         palette="colorblind")
        
        box.set(
            xlabel=x_lab, 
            ylabel=y_lab)
        
        box.set_title(main_title)
        box.set_xticklabels(x_ticks)            
        
        plt.show()
        
        
        
        




