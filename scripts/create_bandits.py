# -*- coding: utf-8 -*-

'''
The code is a Python script that creates bandit tasks, which are saved in zip files 
to be later tested on trained RNN agents
'''

# Import necessary modules
from classes.bandits import fixed_bandit_class as fbc

# Bandit settings

# Type of bandit (str)
BANDIT_TYPE = 'restless' # 'stationary', 'restless', 'meta_volatility', 'daw_et_al_2006'
# Number of bandit arms (int)
ARMS = 2 
# Number of trials (int)
NUM_STEPS = 100
# Number of runs to create with certain bandit specifications (int)
NUM_RUNS = 1 
# Number of reward instantiations given fixed reward probability, only relevant for binary rewards (int)
NUM_RINS = 1
# Standard deviation of Gaussian noise (float)
NOISE_SD = 0.1
# Should reward (probabilities) of arms be dependent? (bool)
DEPENDANT = False 
# If reward_type is 'binary': non-rewards are negative, if reward_type is 'continuous': rewards are mean-centered (bool)
PUNISH = True
# Type of rewards ('binary' or 'continuous')
REWARD_TYPE = 'continuous'
# Path to save bandit data as zip files (str)
PATH_TO_SAVE_BANDITS = 'data/intermediate_data/fixed_bandits/'

# initialize fixed bandit class
fixed_bandit = fbc.CreateBandit(  
    bandit_type = BANDIT_TYPE,
    arms = ARMS, 
    num_steps = NUM_STEPS, 
    num_runs = NUM_RUNS, 
    num_rins = NUM_RINS, 
    noise_sd = NOISE_SD, 
    dependant = DEPENDANT, 
    punish = PUNISH,
    reward_type = REWARD_TYPE,
    path_to_save_bandits = PATH_TO_SAVE_BANDITS)

# generate bandit tasks
fixed_bandit.generate_and_save_bandits()

