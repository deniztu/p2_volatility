# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:31:34 2021

@author: Deniz
"""

import numpy as np

from helpers import feather_class
from helpers import dot2_

feather_cl = feather_class(feather_file_name = '{}_nh_48_lr_0_0001_n_u_p_0_5_ew_lin_vw_0_5_dr_0_5_res_d_f_p_{}_rt_con_a_4_n_300_te_50000_id_{}_test_b_daw_p_{}')


#feather_file_name = 'all_{}_{}_test_runs_train_sd_.1_id_9_test_sd_0_32'

TRAIN_SD = ['0_1']
IDS = 20
RUNS = 1
RINS = 0
# TEST_SDS = np.arange(.02, .34, .02)
TEST_SDS = [1, 2, 3]
REWARD_TYPE = ''
RNN_TYPE = 'lstm2_a2c'
IS_NOISE = False


for train_sd in TRAIN_SD:
    for id_ in range(0,IDS):
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