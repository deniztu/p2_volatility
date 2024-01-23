# -*- coding: utf-8 -*-
"""
This is a collection of helper functions

@author: Deniz
"""
import os
import zipfile
import pickle

'''
Class to manage zip files
'''

class zip2csv():
    def __init__(self, path_to_data, zip_file_name):
        self.zip_file_name = zip_file_name
        self.full_zip_file_name = path_to_data + '\{}'.format(self.zip_file_name)
    
    def extract_file(self, file_name):
        # extract a single file
        with zipfile.ZipFile(self.full_zip_file_name, 'r') as my_zip:
            my_zip.extract(file_name)
    
    def delete_file(self, file_name):
        os.remove(file_name)
        
    def extract_all_files(self):
        # extract a all files
        with zipfile.ZipFile(self.full_zip_file_name, 'r') as my_zip:
            my_zip.extractall()
    
    def delete_all_files(self):
        if self.zip_file_name.endswith('.zip'):
            zip_file_names = self.zip_file_name[:-4]+'*'
        os.system('cmd /c "del {}"'.format(zip_file_names))
        
        
"""
function converts float with dot to string with _
"""

def dot2_(num, is_lr = False):
    
    if isinstance(num, str):
        return(num[0:3])
    
    try: 
        res = str(round(num, 2)).replace('.', '_')
    except: 
        res = 'n'
    # do not round if it is a learning_rate
    if is_lr:
        res = str(num).replace('.', '_')
    return(res)

"""
function to transfer files between R and python with feather
"""

class feather_class():
    def __init__(self, path_to_test_runs = 'data/rnn_raw_data/'
                 , feather_file_name = 'all_{}_test_runs_train_sd_{}_id_{}_test_sd_{}'
                 , path_of_feather_file = 'data/intermediate_data/modeling/preprocessed_data_for_modeling'):
        
        self.path_to_test_runs = path_to_test_runs
        self.feather_file_name = feather_file_name
        self.path_of_feather_file = path_of_feather_file
        
    def create_feather(self, rnn_type, is_noise, train_sd, id_, test_sd_str, test_sd_num, reward_type):
        
        # pdb.set_trace()
        
        file_name = self.feather_file_name.format(rnn_type, train_sd, id_, test_sd_str)
        # complete_string = self.path_to_test_runs + '\{}'.format(file_name)
                
        # os.chdir(self.path_of_feather_file)
        
        all_test_runs = pickle.load(open(self.path_to_test_runs + file_name, 'rb'))
        
        # os.chdir('\..\..')
        
        #pdb.set_trace()
        
        # mult_df = all_test_runs.query('rnn_test_sd == {}'.format(test_sd_num))
        
        mult_df = all_test_runs.reset_index()
        

        # indices = mult_df.index.names
        
        # indices = np.array(indices)
        
        final_df = mult_df.reset_index()
        
        # feather_file_name = '{}_n_{}_rt_{}_train_sd_{}_id_{}_test_sd_{}'.format(rnn_type, str(is_noise).lower()[0], reward_type, train_sd, id_, test_sd_str)
        
        final_df.to_feather(self.path_of_feather_file + '/{}'.format(file_name))
        
    def delete_feather(self, rnn_type, is_noise, train_sd, id_, test_sd_str, run, rin, reward_type):
        
        feather_file_name = '{}_n_{}_rt_{}_train_sd_{}_id_{}_test_sd_{}'.format(rnn_type, str(is_noise).lower()[0], reward_type, train_sd, id_, test_sd_str)
        
        os.remove(self.path_of_feather_file + '/{}'.format(feather_file_name))
            
# test = feather_class()        

# test.create_feather('meta_volatility', '0', '0_02', run = 0, rin = 0, reward_type = 'binary')
            
            
            




