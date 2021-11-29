# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:31:34 2021

@author: Deniz
"""

import numpy as np

from helpers import feather_class
from helpers import dot2_

feather_cl = feather_class()

TRAIN_SD = ['meta_volatility', '.05','.1']
IDS = 10
RUNS = 10
RINS = 0
TEST_SDS = np.arange(.02, .34, .02)
REWARD_TYPE = 'continuous'
RNN_TYPE = 'rnn'
IS_NOISE = False

for train_sd in TRAIN_SD:
    for id_ in range(IDS):
        for test_sd in TEST_SDS:
            for run in range(RUNS):
                for rin in range(1):
                    
                    sd_ = dot2_(test_sd)
                    
                    
                    
                    # feather python file
                    feather_cl.create_feather(rnn_type = RNN_TYPE
                                              , is_noise = IS_NOISE
                                              , train_sd = train_sd
                                              , id_ = id_
                                              , test_sd_str = sd_
                                              , test_sd_num = test_sd
                                              , reward_type = REWARD_TYPE)