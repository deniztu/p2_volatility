# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:54:53 2021

@author: Deniz
"""
import numpy as np

from classes.pca_handler import pca_handler
from helpers import dot2_

pca_class = pca_handler()    


for id_ in range(0,10):
    for sd in np.arange(0.02, 0.34, 0.02):
        
        sd_ = dot2_(sd)
        
        pca_class.save_pca_to_pickle(file_name = 'all_continuous_test_runs_train_sd_meta_volatility_id_{}_test_sd_{}'.format(id_, sd_))


# all_continuous_test_runs_train_sd_.1_id_9_test_sd_0_32 -> CHECK

# all_continuous_test_runs_train_sd_.05_id_8_test_sd_0_02 -> CHECK

# all_continuous_test_runs_train_sd_meta_volatility_id_7_test_sd_0_3 -> CHECK