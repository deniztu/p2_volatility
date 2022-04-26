# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 07:07:33 2021

@author: Deniz
"""
'''
class to save principal components from the network to a pickle file
'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import pickle

class pca_handler():
    def __init__(self, path_to_rnn_raw_data = 'data/rnn_raw_data/' 
                  , path_to_save_pca = 'data/intermediate_data/pca/'):
        
        self.path_to_rnn_raw_data = path_to_rnn_raw_data
        self.path_to_save_pca = path_to_save_pca
        
    def save_pca_to_pickle(self, file_name, num_runs = 10):
        
        # load rnn raw data
        df = pickle.load(open( self.path_to_rnn_raw_data + file_name, "rb" ) )
        
        # get hidden neuron activation
        filter_col = [col for col in df if col.startswith('rnn_state')]
        
        # define colnames for pca's
        colnames = []
        for col in range(0,len(filter_col)):
            colnames.append('PC'+str(col+1))
        
        # create list of dfs for multindexing
        df_list = []
        
        # create list of pca objects for each run
        pca_object_per_run = []
        
        # apply pca for each run
        for run in range(num_runs):
             
            # hierarchical indexing, run is 4th position (see df)   
            df_run = df.loc[:,:,:,run,:][filter_col]
            
            # instantiate PCA class
            pca = PCA()
            
            # standardize hidden neuron activation
            x = StandardScaler().fit_transform(df_run)
            
            # calculate pc's
            pcs = pca.fit_transform(x)
            
            # save pc's in df
            pc_df = pd.DataFrame(data = pcs, columns = colnames)
            
            # add columns later used as index
            pc_df['rnn_type'] = df.index[0][0]
            pc_df['rnn_id'] = df.index[0][1]
            pc_df['rnn_test_sd'] = df.index[0][2]
            pc_df['run'] = run
            pc_df['reward_instance'] = df.index[0][4]
        
            # append to df_list for hierarchical df later
            df_list.append(pc_df)
            
            # append to pca object of run to list
            pca_object_per_run.append(pca)
        
        # create list with names of the index of the multiindex df
        multiindex_list = ['rnn_type', 'rnn_id', 'rnn_test_sd', 'run', 'reward_instance']
        
        # concat df_list rowwise
        all_dfs = pd.concat(df_list)
        
        # make all_dfs a muliindex df
        mult_ind_df = all_dfs.set_index(multiindex_list)
        
        # save pc_df and pca to pickle file
        
        pca_file_name = 'pca_{}.pickle'.format(file_name)
        
        my_pca_dict = {'pca_mult_ind_df': mult_ind_df, 'file_name': file_name, 'pca_object_per_run': pca_object_per_run}
        
        file_to_store = open(self.path_to_save_pca + pca_file_name, "wb")
        
        pickle.dump(my_pca_dict, file_to_store)
        
        file_to_store.close()





'''
Test first
'''

# pca_class = pca_handler()    

# pca_class.save_pca_to_pickle(file_name = 'all_continuous_test_runs_train_sd_meta_volatility_id_9_test_sd_0_32')


# path_to_save_pca = 'data/intermediate_data/pca/'
# path_to_rnn_raw_data = 'data/rnn_raw_data/'
# file_name = 'all_continuous_test_runs_train_sd_meta_volatility_id_9_test_sd_0_32'
# num_runs = 10

# # load rnn raw data
# df = pickle.load(open( path_to_rnn_raw_data + file_name, "rb" ) )

# # get hidden neuron activation
# filter_col = [col for col in df if col.startswith('rnn_state')]

# # define colnames for pca's
# colnames = []
# for col in range(0,len(filter_col)):
#     colnames.append('PC'+str(col+1))

# # create list of dfs for multindexing
# df_list = []

# # apply pca for each run
# for run in range(num_runs):
     
#     # hierarchical indexing, run is 4th position (see df)   
#     df_run = df.loc[:,:,:,run,:][filter_col]
    
#     # instantiate PCA class
#     pca = PCA()
    
#     # standardize hidden neuron activation
#     x = StandardScaler().fit_transform(df_run)
    
#     # calculate pc's
#     pcs = pca.fit_transform(x)
    
#     # save pc's in df
#     pc_df = pd.DataFrame(data = pcs, columns = colnames)
    
#     # add columns later used as index
#     pc_df['rnn_type'] = df.index[0][0]
#     pc_df['rnn_id'] = df.index[0][1]
#     pc_df['rnn_test_sd'] = df.index[0][2]
#     pc_df['run'] = run
#     pc_df['reward_instance'] = df.index[0][4]

#     # append to df_list for hierarchical df later
#     df_list.append(pc_df)

# # create list with names of the index of the multiindex df
# multiindex_list = ['rnn_type', 'rnn_id', 'rnn_test_sd', 'run', 'reward_instance']

# # concat df_list rowwise
# all_dfs = pd.concat(df_list)

# # make all_dfs a muliindex df
# mult_ind_df = all_dfs.set_index(multiindex_list)

# # save pc_df and pca to pickle file

# pca_file_name = 'pca_{}.pickle'.format(file_name)

# my_pca_dict = {'pca_mult_ind_df': mult_ind_df, 'file_name': file_name}

# file_to_store = open(path_to_save_pca + pca_file_name, "wb")

# pickle.dump(my_pca_dict, file_to_store)

# file_to_store.close()


#--- CUT

# standardize hidden neuron activation
# x = StandardScaler().fit_transform(df[filter_col])

# # calculate pc's
# pcs = pca.fit_transform(x)

# # save pc's in df
# pc_df = pd.DataFrame(data = pcs,
#                            columns = colnames)

