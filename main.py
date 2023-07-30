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

# =============================================================================
# Cell Type: LSTM | Entropy: linear | Noise: None | Alg: A2C 
# Cell Type: LSTM | Entropy: 0         | Noise: None | Alg: A2C 
# Cell Type: LSTM | Entropy: 0.05     | Noise: None | Alg: A2C 
# =============================================================================




daw_walks = ['classes/bandits/Daw2006_payoffs1.csv',
             'classes/bandits/Daw2006_payoffs2.csv',
             'classes/bandits/Daw2006_payoffs3.csv']

entropies = [0.05, 'linear']


for e in entropies:
    
    for id_ in range(9, 30):    
    
            train_mab = bc.bandit(bandit_type = 'restless'
                                , arms = 4
                                , num_steps = 300
                                , reward_type = 'continuous'
                                , noise_sd = 0.1
                                , punish = True)
            
            nnet = nn.neural_network(bandit = train_mab
                                , noise = 'update-dependant'
                                , discount_rate = 0.5
                                , value_loss_weight= 0.5
                                , entropy_loss_weight = 0.05#e
                                , rnn_type = 'rnn'
                                , noise_parameter = 0.5
                                , learning_algorithm = 'reinforce'
                                , n_iterations = 50000
                                , model_id= 15
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
    

    
    
