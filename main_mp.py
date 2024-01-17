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

import concurrent.futures
import time

# daw_walks = ['classes/bandits/Daw2006_payoffs1.csv',
#              'classes/bandits/Daw2006_payoffs2.csv',
#              'classes/bandits/Daw2006_payoffs3.csv']

def tf_function(id_):
    
    
#for id_ in range(16, 30):
    
    #entropies = [0]
    num_hidden_units = [80]
    for nh in num_hidden_units:
        
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
                            , entropy_loss_weight = 0
                            , rnn_type = 'lstm2'
                            , noise_parameter = 0.5
                            , learning_algorithm = 'a2c'
                            , n_iterations = 50000
                            , model_id= id_
                            , n_hidden_neurons = nh)
        
        # train the rnn
        nnet.train()
        
        # reset the rnn
        nnet.reset()
        
            
        #for daw_walk in range(3):

            # daw walk
            # test_mab = daw_walks[daw_walk]
            
            # nnet.test(bandit = test_mab, bandit_param_range = [daw_walk+1], n_runs = 1)
    
        # test the rnn
        nnet.test(bandit = 'fixed_res_rt_con_p_{}_a_4_n_300_run_{}.zip',
                  bandit_param_range = [0.1],
                  n_runs = 30)
        

if __name__ == '__main__':

    ids =range(24, 30)

    start = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(tf_function, ids)
    

    finish = time.perf_counter()
    
    print(f'Finished in {round(finish-start, 2)} second(s)')


        
        
