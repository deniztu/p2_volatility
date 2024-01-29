# -*- coding: utf-8 -*-

import numpy as np

class LoadDawBandit:
    """
    Class to load bandit data where the reward generating process is according to Daw et al. 2006

    Parameters
    ----------
    fixed_bandit : pandas.DataFrame
    
        DataFrame containing following columns:
            
        - Index 0: Index column
        - Index 1 - n_arms: rewards

    Attributes
    ----------
    fixed_bandit : pandas.DataFrame
    
        DataFrame containing following columns:
            
        - Index 0: Index column
        - Index 1 - n_arms: rewards
        
    arms : int
        Number of bandit arms (excluding the first column).
    num_steps : int
        Number of trials (default: 300).
    bandit_type : str
        Type of bandit ('daw_et_al_2006').
    bandit_parameter : str
        Bandit parameter ('n').

    Methods
    -------
    generate_task()
        Generate task where the reward generating process is according to Daw et al. 2006
    """
    def __init__(self, fixed_bandit):
        """
        Initialize the LoadDawBandit instance.

        Parameters
        ----------
        fixed_bandit : pandas.DataFrame
        
            DataFrame containing following columns:
                
            - Index 0: Index column
            - Index 1 - n_arms: rewards
        """
        self.fixed_bandit = fixed_bandit
        self.arms = int(len(self.fixed_bandit.columns) - 1)
        self.num_steps = 300
        self.bandit_type = 'daw_et_al_2006'
        self.bandit_parameter = 'n'

    def generate_task(self):
        """
        Generate task where the reward generating process is according to Daw et al. 2006

        Returns
        -------
        Tuple
            Tuple containing mean-centered pay-off array and non-mean-centered rewards array.
            
        Notes
        -----
        The method scales rewards from 0-100 to 0-1!
        """
        # Convert DataFrame to np.array and omit the first column
        arr = self.fixed_bandit.to_numpy()[:self.num_steps, 1:]

        # Get rewards
        rewards = arr[:, :self.arms]

        # Scale rewards between 0-1 (max points 100)
        rewards = rewards / 100

        # Mean center rewards (commented to test mean center effect on exploration)
        centered_pay_off_arr = rewards - np.mean(rewards)

        return centered_pay_off_arr, rewards




# class load_daw_bandit:

#     def __init__(self, fixed_bandit):

        
#         self.fixed_bandit = fixed_bandit
        
#         # get N_ARMS (omit first column)
#         self.arms = int(len(self.fixed_bandit.columns)-1)
#         # set 300 trials
#         self.num_steps = 300
#         # beware not dynamic
#         self.bandit_type = 'daw_et_al_2006'
#         # beware not dynamic
#         self.bandit_parameter = 'n'

#     def generate_task(self):

    
#         # convert df to np.array and omit first column
#         arr = self.fixed_bandit.to_numpy()[:self.num_steps,1:]

#         # get rewards
#         rewards = arr[:,:self.arms]
        
#         # scale rewards between 0-1 (max points 100)
#         rewards = rewards/100

#         # mean center rewards (commented to test mean center effect on exploration)
#         centered_pay_off_arr = rewards - np.mean(rewards)        
        
#         return(centered_pay_off_arr, rewards)



