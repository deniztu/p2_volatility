import pandas as pd
import numpy as np
import os

# use bandit_class to create bandits in create_bandit
from classes.bandits import bandit_class as bc 
import zipfile
from helpers import dot2_

class load_bandit:
    '''
    Class loads presaved bandits
    
    Input:
        fixed_bandit: pd.DataFrame with rewards ('reward_a') 
                      and reward probability ('p_rew_a')
                      a is index 1 to N_Arms
        
    Output: 
        rewards: numpy.ndarray, shape[num_steps, arms]
        p_rew: numpy.ndarray, probability of reward per arm
            
    '''
    def __init__(self, fixed_bandit):
        

        self.fixed_bandit = fixed_bandit
        
        # get N_ARMS from column names
        self.arms = int(self.fixed_bandit.columns[-1][-1])
        
        # get num_steps from column names
        self.num_steps = int(self.fixed_bandit.shape[0])
        
        # beware not dynamic
        self.bandit_type = 'restless'
        
        # beware not dynamic
        self.bandit_parameter = 'this changes'
        
        # beware not dynamic
        self.reward_type = 'binary'
        
            
    def generate_task(self):
        
        # convert df to np.array and omit first column
        arr = self.fixed_bandit.to_numpy()[:,1:]
        
        # get rewards
        rewards = arr[:,:self.arms]
        
        # get reward probs
        r_probs = arr[:,self.arms:]
            
        return rewards, r_probs
    
class create_bandit:
    '''
    Class creates bandits and saves them into a .zip file
    
    Input: 
        bandit_type: string, either:
            - 'stationary'
            - 'restless'
            - 'meta_volatility'
            - 'daw_et_al_2006'
            
        arms: int, number of bandit arms
        num_steps: int, number of trials
        dependant: bool, should reward probs of arms be dependant
        reward_type: 'continuous' or 'binary', note if 'continuous' rewards are mean centered!
        num_runs: int, number of reward schedule drawings (runs) with given reward_rate or noise_sd
        num_rins: int, number of reward instances from a given run (only necessary in binary bandits)
        punish: bool, should non-rewards be negative
        
        if bandit_type == 'stationary':
            reward_rate: float, probability of observing reward
        
        if bandit_type == 'restless':
            noise_sd: float, sd of the gaussian noise (mu = 0, sd = noise_sd)
            
        if bandit_type == 'meta_volatility':
            restless bandits with noise_sd drawn from [0.02 - 0.2]
        
    Output:
        zip-file with bandit runs and optionally reward instances (rins)
            
    '''
    def __init__(self
                 , bandit_type
                 , arms
                 , num_steps
                 , num_runs
                 , num_rins
                 , noise_sd = 2.8
                 , reward_rate = None
                 , dependant = False
                 , punish = True
                 , reward_type = 'binary'
                 , path_to_save_bandits = 'data/intermediate_data/fixed_bandits/'):
        
        self.bandit = bc.bandit(bandit_type 
                , arms 
                , num_steps
                , noise_sd
                , reward_rate
                , dependant
                , punish
                , reward_type)
        
        self.num_runs = num_runs
        self.num_rins = num_rins
        

        

        
        for run in range(num_runs):
            
            # create zip file
            zip_name = 'fixed_{}_rt_{}_p_{}_a_{}_n_{}_run_{}.zip'.format(self.bandit.bandit_type[0:3]
                                                                        , self.bandit.reward_type[0:3]
                                                                        , dot2_(self.bandit.bandit_parameter)
                                                                        , self.bandit.arms
                                                                        , self.bandit.num_steps
                                                                        , str(run)).lower()
                    
            # open zip file to save test runs
            with zipfile.ZipFile(path_to_save_bandits+'{}'.format(zip_name), 'w', compression = zipfile.ZIP_DEFLATED) as my_zip:
                                
                # generate the bandit run
                _, r_probs = self.bandit.generate_task()
                
                if reward_type == 'binary':
                    
                    for rin in range(num_rins):
                        
                        # generate reward instance
                        rewards      = np.zeros([num_steps, arms])          
                
                        # perform bernoulli trials with reward probs
                        random_numb  = np.random.rand(num_steps, arms)
                        rewards = (r_probs > random_numb) * 1.
                        
                        if punish:
                            rewards = 2*rewards-1
                            
                        # collect rewards and reward probability
                        data = np.hstack((rewards, r_probs))
                        data.shape
                        
                        # create dataframe
                        p_rew_cols = ['p_rew_'+ str(i+1) for i in range(arms)]
                        reward_cols = ['reward_'+ str(i+1) for i in range(arms)]
                        df = pd.DataFrame(data, columns= reward_cols + p_rew_cols)
                        
    
                        file_name = zip_name.replace('.zip', '') +'_rin_{}.csv'.format(str(rin)).lower()
                                                                                       
                        # create csv from dataframe
                        df.to_csv(file_name)
                        # write csv to zip file
                        my_zip.write(file_name)
                        # delete csv file
                        os.remove(file_name) 
                        
                if reward_type == 'continuous' and num_rins == 1:
                    
                    # mean center rewards
                    rewards = r_probs - np.mean(r_probs)
                    
                    # collect rewards and reward probability
                    data = np.hstack((rewards, r_probs))
                    
                    # create dataframe
                    p_rew_cols = ['p_rew_'+ str(i+1) for i in range(arms)]
                    reward_cols = ['reward_'+ str(i+1) for i in range(arms)]
                    df = pd.DataFrame(data, columns= reward_cols + p_rew_cols)
                        
                    # pdb.set_trace()
                    rin = 0 # ugly quick fix reward instance = 1
                    file_name = zip_name.replace('.zip', '') +'_rin_{}.csv'.format(str(rin)).lower()
                                                                                   
                    # create csv from dataframe
                    df.to_csv(file_name)
                    # write csv to zip file
                    my_zip.write(file_name)
                    # delete csv file
                    os.remove(file_name)
            
                else:
                    raise ValueError('this functionality is not implemented yet')
                    

                    
                    
                
                
                    
                    
            
        

        
        

        
        
