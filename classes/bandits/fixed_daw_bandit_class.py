# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:49:31 2021

@author: Deniz
"""

import pandas as pd
import numpy as np

class load_daw_bandit:
    '''
    
    '''
    def __init__(self, fixed_bandit):
        self.fixed_bandit = fixed_bandit
        
        # get N_ARMS (omit first column)
        self.arms = int(len(self.fixed_bandit.columns)-1)
        # set 300 trials
        self.num_steps = 300
        # beware not dynamic
        self.bandit_type = 'daw_et_al_2006'
        # beware not dynamic
        self.bandit_parameter = 'n'

    def generate_task(self):
    
        # convert df to np.array and omit first column
        arr = self.fixed_bandit.to_numpy()[:self.num_steps,1:]

        # get rewards
        rewards = arr[:,:self.arms]
        
        # scale rewards between 0-1 (max points 100)
        rewards = rewards/100

        # mean center rewards (commented to test mean center effect on exploration)
        centered_pay_off_arr = rewards - np.mean(rewards)
        #pay_off_arr = rewards
        
        
        return(centered_pay_off_arr, rewards)

# TEST
# test = load_daw_bandit(df)
# sc_rews, rews = test.generate_task()


