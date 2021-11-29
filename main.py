from classes.bandits import fixed_bandit_class as fbc
from classes.bandits import bandit_class as bc
from classes.neural_networks import network_class_LSTM_v2_Deniz_IP2_AC3 as nn
# from neural_networks import network_class as nn
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

#%% Train the rnn

bandit_types = ['.1']

reward_types = ['continuous']

for id_ in range(10):
    for isnoise in [False]:
        for reward_type in reward_types:
            for bandit_type in bandit_types:
            
                # intitialise training bandit
                
                if bandit_type == '.05':
                    this_bandit_type = 'restless'
                    sd = .05
                    
                if bandit_type == '.1':
                    this_bandit_type = 'restless'
                    sd = .1
                    
                if bandit_type == 'meta_volatility':
                    this_bandit_type = bandit_type
                    sd = None
                    
                
                train_mab = bc.bandit(bandit_type = this_bandit_type
                                    , arms = 4 
                                    , num_steps = 300
                                    , reward_type = reward_type
                                    , noise_sd = sd)
                        
                # intitialise rnn class
                nnet = nn.neural_network(bandit = train_mab
                                    , noise = False
                                    , entropy_scaling=0
                                    , weber_fraction=0.5
                                    , n_iterations = 50000
                                    , model_id= id_
                                    , train_sd = bandit_type)
                # train the rnn
                nnet.train()
                
                # reset the rnn
                nnet.reset()
                
                # sd_range = np.arange(0.02, 0.34, 0.02)        
                
                # for sd_ in sd_range:
                
                #     fbc.create_bandit(bandit_type = 'restless'
                #                         , arms = 4 
                #                         , num_steps = 300
                #                         , reward_type='continuous'
                #                         , num_runs = 10
                #                         , num_rins = 1
                #                         , noise_sd = sd_)
                
                # test_mab = 'fixed_res_rt_con_p_{}_a_4_n_300_run_{}.zip'
                
                # # test the rnn
                # nnet.test(n_replications = 10,
                #           num_rins = 1,
                #           bandit = test_mab,
                #           bandit_param_range = np.arange(0.02, 0.34, 0.02),
                #           use_fixed_bandits = True,
                #           reward_type = 'continuous')
                
                # nnet.test(n_replications = 10,
                # num_rins = 1,
                # bandit = test_mab,
                # bandit_param_range = np.arange(0.02, 0.34, 0.02),
                # use_fixed_bandits = True,
                # reward_type = 'continuous')
                

                
# # flatten final_df_list as it is list of lists
# flat_list = [item for sublist in final_df_list for item in sublist]

# # create list with names of the index of the multiindex df
# multiindex_list = ['rnn_type', 'rnn_id', 'rnn_test_sd', 'run', 'reward_instance']

# # concat df_list rowwise
# all_dfs = pd.concat(flat_list)

# # make all_dfs a muliindex df
# mult_ind_df = all_dfs.set_index(multiindex_list)

# # pickle the file
# filename = 'all_binary_test_runs'
# outfile = open(filename,'wb')

# pickle.dump(mult_ind_df, outfile)

# outfile.close()


        
        
    
    
    
    
    
