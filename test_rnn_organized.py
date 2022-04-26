# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:29:28 2022

@author: Deniz
"""

from classes.bandits import fixed_bandit_class as fbc
from classes.bandits import bandit_class as bc
from classes.neural_networks import network_class_organized as nn
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

train_mab = bc.bandit(bandit_type = 'restless'
                    , arms = 4
                    , num_steps = 300
                    , reward_type = 'continuous'
                    , noise_sd = 0.1
                    , dependant = False)

nnet = nn.neural_network(bandit = train_mab
                    , noise = 'none'
                    , entropy_loss_weight = 0.05
                    , value_loss_weight= 0.5
                    , rnn_type = 'lstm'
                    , learning_algorithm = 'a2c'
                    , n_iterations = 50000
                    , model_id= 0)

# train the rnn
# nnet.train()

# reset the rnn
nnet.reset()

test_mab = 'fixed_res_rt_con_p_{}_a_4_n_300_run_{}.zip'

nnet.test(bandit = test_mab, bandit_param_range = [0.1], n_runs = 10)

df = pickle.load(open('lstm_a2c_nh_48_lr_0_0001_n_n_p_0_ew_0_05_vw_0_5_dr_0_5_res_d_f_p_0_1_rt_con_a_4_n_300_te_50000_id_0_test_b_res_p_0_1', 'rb'))
