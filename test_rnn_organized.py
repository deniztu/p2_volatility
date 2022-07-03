# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:29:28 2022

@author: Deniz
"""

from classes.bandits import fixed_bandit_class as fbc
from classes.bandits import bandit_class as bc
from classes.neural_networks import network_class_organized_lstm_cell as nn
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle


daw_walks = ['classes/bandits/Daw2006_payoffs1.csv',
             'classes/bandits/Daw2006_payoffs2.csv',
             'classes/bandits/Daw2006_payoffs3.csv']

for id_ in range(4,20):    

        train_mab = bc.bandit(bandit_type = 'stationary'
                            , arms = 2
                            , num_steps = 50
                            , reward_type = 'binary'
                            , reward_rate = 0.9
                            , dependant = True)
        
        nnet = nn.neural_network(bandit = train_mab
                            , noise = 'update-dependant'
                            , value_loss_weight= 0.5
                            , entropy_loss_weight = 0
                            , rnn_type = 'lstm2'
                            , noise_parameter = 0.5 
                            , learning_algorithm = 'a2c'
                            , n_iterations = 20000
                            , model_id= 999
                            , n_hidden_neurons = 48)
        
        # train the rnn
        nnet.train()
        
        # reset the rnn
        nnet.reset()
        
        #test_mab = 'fixed_res_rt_con_p_{}_a_4_n_300_run_{}.zip'
        
        for daw_walk in range(3):

            # daw walk
            test_mab = daw_walks[daw_walk]
            
            nnet.test(bandit = test_mab, bandit_param_range = [daw_walk+1], n_runs = 1)

# df = pickle.load(open('data/rnn_raw_data/rnn_rei_nh_48_lr_0_0001_n_n_p_0_ew_0_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_7_test_b_res_p_0_2', 'rb'))


# df.index

# # calculate rank for row
# ranks = df[['p_rew_1', 'p_rew_2', 'p_rew_3', 'p_rew_4']].rank(axis=1, ascending=False)

# # get chosen rank 
# ch_rank = [[r1, r2, r3, r4][ch] for ch, r1, r2, r3, r4 in zip(df['choice'], ranks['p_rew_1'], ranks['p_rew_2'], ranks['p_rew_3'], ranks['p_rew_4'])]

# # append chosen rank to df
# df['chosen_rank'] = ch_rank

# x = df.groupby(['rnn_id','test_sd']).chosen_rank.mean()

# x.groupby(['rnn_id','test_sd']).mean()


# # rename cols
# df.rename_axis(index={'rnn_test_sd': "test_sd"}, inplace=True)


