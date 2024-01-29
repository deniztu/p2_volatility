import pandas as pd
import numpy as np
import os

# use bandit_class to create bandits in create_bandit
from classes.bandits import bandit_class as bc 
import zipfile
from helpers import dot2_

class LoadBandit:
    """Class to load a bandit based on the provided data.

    Parameters
    ----------
    fixed_bandit : pandas.DataFrame
        
        DataFrame containing following columns:
            
        - Index 0: Index column
        - Index 1 - n_arms: rewards with column names e.g. 'reward_1' to 'reward_4' (if n_arms = 4)
        - Index n_arms+1 - last_column: probability of reward with column names e.g., p_rew_1 to p_rew_4 (if n_arms = 4)

    Attributes
    ----------
    arms : int
        Number of bandit arms
    
    num_steps : int 
        Number of trials
    
    bandit_type : str 
        Type of bandit (not dynamic, set to 'restless')
    
    bandit_parameter : str
        Bandit parameter (not dynamic, set to 'null')
    
    reward_type : str
        Type of rewards (not dynamic, set to 'binary')

    Methods:
    -------
    generate_task
        Generate bandit task based on the provided data.

    """

    def __init__(self, fixed_bandit):
        

        self.fixed_bandit = fixed_bandit
        
        # get N_ARMS from column names
        self.arms = int(self.fixed_bandit.columns[-1][-1])
        
        # get num_steps from column names
        self.num_steps = int(self.fixed_bandit.shape[0])
        
        # beware not dynamic
        self.bandit_type = 'restless'
        
        # beware not dynamic
        self.bandit_parameter = 'null'
        
        # beware not dynamic
        self.reward_type = 'binary'
        
            
    def generate_task(self):
        
        '''
        Generate bandit task based on the provided data.

        Returns:
        - Tuple: Tuple containing rewards array and reward probability array.
        '''
        
        # convert df to np.array and omit first column
        arr = self.fixed_bandit.to_numpy()[:,1:]
        
        # get rewards
        rewards = arr[:,:self.arms]
        
        # get reward probs
        r_probs = arr[:,self.arms:]
            
        return rewards, r_probs


class CreateBandit:
    """
    Class to create and save bandit data.

    Parameters
    ----------
    bandit_type : str
        Type of bandit ('stationary', 'restless', 'meta_volatility', 'daw_et_al_2006').
    arms : int
        Number of bandit arms.
    num_steps : int
        Number of trials.
    num_runs : int
        Number of runs.
    num_rins : int
        Number of reward instances.
    noise_sd : float, optional
        Standard deviation of Gaussian noise (default: 2.8).
    reward_rate : float, optional
        Probability of observing reward for stationary bandit (default: None).
    dependant : bool, optional
        Should reward probabilities of arms be dependent (default: False).
    punish : bool, optional
        If reward_type is 'binary': non-rewards are negative, if reward_type is 'continuous': rewards are mean-centered (default: True).
    reward_type : str, optional
        Type of rewards ('binary' or 'continuous', default: 'binary').
    path_to_save_bandits : str, optional
        Path to save bandit data (default: 'data/intermediate_data/fixed_bandits/').

    Attributes
    ----------
    bandit : Bandit 
        Bandit instance from bandit_class
    num_runs : int
        Number of runs.
    num_rins : int
        Number of reward instances.
    path_to_save_bandits : str
        Path to save bandit data (default: 'data/intermediate_data/fixed_bandits/').
    reward_type : str, optional
        Type of rewards ('binary' or 'continuous', default: 'binary').

    Methods
    -------
    generate_and_save_bandits()
        Generate and save bandit data.
    """
    def __init__(self, bandit_type, arms, num_steps, num_runs, num_rins, noise_sd=2.8,
                 reward_rate=None, dependant=False, punish=True, reward_type='binary',
                 path_to_save_bandits='data/intermediate_data/fixed_bandits/'):
        """
        Initialize the CreateBandit instance.

        Parameters
        ----------
        bandit_type : str
            Type of bandit ('stationary', 'restless', 'meta_volatility', 'daw_et_al_2006').
        arms : int
            Number of bandit arms.
        num_steps : int
            Number of trials.
        num_runs : int
            Number of runs.
        num_rins : int
            Number of reward instances.
        noise_sd : float, optional
            Standard deviation of Gaussian noise (default: 2.8).
        reward_rate : float, optional
            Probability of observing reward for stationary bandit (default: None).
        dependant : bool, optional
            Should reward probabilities of arms be dependent (default: False).
        punish : bool, optional
            If reward_type is 'binary': non-rewards are negative, if reward_type is 'continuous': rewards are mean-centered (default: True).
        reward_type : str, optional
            Type of rewards ('binary' or 'continuous', default: 'binary').
        path_to_save_bandits : str, optional
            Path to save bandit data (default: 'data/intermediate_data/fixed_bandits/').

        """
        self.bandit = bc.Bandit(bandit_type, arms, num_steps, noise_sd, reward_rate,
                                dependant, punish, reward_type)
        self.num_runs = num_runs
        self.num_rins = num_rins
        self.path_to_save_bandits = path_to_save_bandits
        self.reward_type = reward_type

    def generate_and_save_bandits(self):
        """
        Generate and save bandit data to zip files to path_to_save_bandits.

        Raises
        ------
        ValueError
            If reward_type is 'continuous' and num_rins is not 1.

        """
        for run in range(self.num_runs):
            # Create zip file
            zip_name = 'fixed_{}_rt_{}_p_{}_a_{}_n_{}_run_{}.zip'.format(
                self.bandit.bandit_type[0:3],
                self.bandit.reward_type[0:3],
                dot2_(self.bandit.bandit_parameter),
                self.bandit.arms,
                self.bandit.num_steps,
                str(run)).lower()

            # Open zip file to save test runs
            with zipfile.ZipFile(self.path_to_save_bandits + '{}'.format(zip_name), 'w', compression=zipfile.ZIP_DEFLATED) as my_zip:
                # Generate the bandit run
                _, r_probs = self.bandit.generate_task()

                if self.reward_type == 'binary':
                    for rin in range(self.num_rins):
                        # Generate reward instance
                        rewards = np.zeros([self.num_steps, self.bandit.arms])

                        # Perform Bernoulli trials with reward probs
                        random_numb = np.random.rand(self.num_steps, self.bandit.arms)
                        rewards = (r_probs > random_numb) * 1.

                        if self.punish:
                            rewards = 2 * rewards - 1

                        # Collect rewards and reward probability
                        data = np.hstack((rewards, r_probs))
                        data.shape

                        # Create dataframe
                        p_rew_cols = ['p_rew_' + str(i + 1) for i in range(self.bandit.arms)]
                        reward_cols = ['reward_' + str(i + 1) for i in range(self.bandit.arms)]
                        df = pd.DataFrame(data, columns=reward_cols + p_rew_cols)

                        file_name = zip_name.replace('.zip', '') + '_rin_{}.csv'.format(str(rin)).lower()

                        # Create CSV from dataframe
                        df.to_csv(file_name)
                        # Write CSV to zip file
                        my_zip.write(file_name)
                        # Delete CSV file
                        os.remove(file_name)

                elif self.reward_type == 'continuous' and self.num_rins == 1:
                    # Mean center rewards
                    rewards = r_probs - np.mean(r_probs)

                    # Collect rewards and reward probability
                    data = np.hstack((rewards, r_probs))

                    # Create dataframe
                    p_rew_cols = ['p_rew_' + str(i + 1) for i in range(self.bandit.arms)]
                    reward_cols = ['reward_' + str(i + 1) for i in range(self.bandit.arms)]
                    df = pd.DataFrame(data, columns=reward_cols + p_rew_cols)

                    rin = 0  # Ugly quick fix reward instance = 1
                    file_name = zip_name.replace('.zip', '') + '_rin_{}.csv'.format(str(rin)).lower()

                    # Create CSV from dataframe
                    df.to_csv(file_name)
                    # Write CSV to zip file
                    my_zip.write(file_name)
                    # Delete CSV file
                    os.remove(file_name)

                else:
                    raise ValueError('This functionality is not implemented yet')

# class CreateBandit:
#     """
#     Class to create and save bandit data.
    
#     Parameters:
#         bandit_type (str): Type of bandit ('stationary', 'restless', 'meta_volatility', 'daw_et_al_2006').
#         arms (int): Number of bandit arms.
#         num_steps (int): Number of trials.
#         num_runs (int): Number of runs.
#         num_rins (int): Number of reward instances.
#         noise_sd (float): Standard deviation of Gaussian noise (default: 2.8).
#         reward_rate (float): Probability of observing reward for stationary bandit (default: None).
#         dependant (bool): Should reward probabilities of arms be dependent (default: False).
#         punish (bool): If reward_type is 'binary': non-rewards are negative, if reward_type is 'continuous': rewards are mean-centered (default: True).
#         reward_type (str): Type of rewards ('binary' or 'continuous', default: 'binary').
#         path_to_save_bandits (str): Path to save bandit data (default: 'data/intermediate_data/fixed_bandits/').
    
#     Attributes:
#         bandit (Bandit): Bandit instance.
#         num_runs (int): Number of runs.
#         num_rins (int): Number of reward instances.
#         path_to_save_bandits (str): Path to save bandit data (default: 'data/intermediate_data/fixed_bandits/').
    
#     Methods:
#         generate_and_save_bandits: Generate and save bandit data.
    
#     """

#     def __init__(self, bandit_type, arms, num_steps, num_runs, num_rins, noise_sd=2.8,
#                  reward_rate=None, dependant=False, punish=True, reward_type='binary',
#                  path_to_save_bandits='data/intermediate_data/fixed_bandits/'):
#         """
#         Initialize the CreateBandit instance.

#         Parameters:
#             bandit_type (str): Type of bandit ('stationary', 'restless', 'meta_volatility', 'daw_et_al_2006').
#             arms (int): Number of bandit arms.
#             num_steps (int): Number of trials.
#             num_runs (int): Number of runs.
#             num_rins (int): Number of reward instances.
#             noise_sd (float): Standard deviation of Gaussian noise (default: 2.8).
#             reward_rate (float): Probability of observing reward for stationary bandit (default: None).
#             dependant (bool): Should reward probabilities of arms be dependent (default: False).
#             punish (bool): If reward_type is 'binary': non-rewards are negative, if reward_type is 'continuous': rewards are mean-centered (default: True).
#             reward_type (str): Type of rewards ('binary' or 'continuous', default: 'binary').
#             path_to_save_bandits (str): Path to save bandit data (default: 'data/intermediate_data/fixed_bandits/').

#         """
#         self.bandit = bc.bandit(bandit_type, arms, num_steps, noise_sd, reward_rate,
#                                 dependant, punish, reward_type)
#         self.num_runs = num_runs
#         self.num_rins = num_rins
#         self.path_to_save_bandits = path_to_save_bandits

#     def generate_and_save_bandits(self):
#         """
#         Generate and save bandit data as zip files to 'path_to_save_bandits'.

#         Raises:
#             ValueError: If reward_type is 'continuous' and num_rins is not 1.

#         """
#         for run in range(self.num_runs):
#             # Create zip file
#             zip_name = 'fixed_{}_rt_{}_p_{}_a_{}_n_{}_run_{}.zip'.format(
#                 self.bandit.bandit_type[0:3],
#                 self.bandit.reward_type[0:3],
#                 dot2_(self.bandit.bandit_parameter),
#                 self.bandit.arms,
#                 self.bandit.num_steps,
#                 str(run)).lower()

#             # Open zip file to save test runs
#             with zipfile.ZipFile(self.path_to_save_bandits + '{}'.format(zip_name), 'w', compression=zipfile.ZIP_DEFLATED) as my_zip:
#                 # Generate the bandit run
#                 _, r_probs = self.bandit.generate_task()

#                 if self.reward_type == 'binary':
#                     for rin in range(self.num_rins):
#                         # Generate reward instance
#                         rewards = np.zeros([self.num_steps, self.bandit.arms])

#                         # Perform Bernoulli trials with reward probs
#                         random_numb = np.random.rand(self.num_steps, self.bandit.arms)
#                         rewards = (r_probs > random_numb) * 1.

#                         if self.punish:
#                             rewards = 2 * rewards - 1

#                         # Collect rewards and reward probability
#                         data = np.hstack((rewards, r_probs))
#                         data.shape

#                         # Create dataframe
#                         p_rew_cols = ['p_rew_' + str(i + 1) for i in range(self.bandit.arms)]
#                         reward_cols = ['reward_' + str(i + 1) for i in range(self.bandit.arms)]
#                         df = pd.DataFrame(data, columns=reward_cols + p_rew_cols)

#                         file_name = zip_name.replace('.zip', '') + '_rin_{}.csv'.format(str(rin)).lower()

#                         # Create CSV from dataframe
#                         df.to_csv(file_name)
#                         # Write CSV to zip file
#                         my_zip.write(file_name)
#                         # Delete CSV file
#                         os.remove(file_name)

#                 elif self.reward_type == 'continuous' and self.num_rins == 1:
#                     # Mean center rewards
#                     rewards = r_probs - np.mean(r_probs)

#                     # Collect rewards and reward probability
#                     data = np.hstack((rewards, r_probs))

#                     # Create dataframe
#                     p_rew_cols = ['p_rew_' + str(i + 1) for i in range(self.bandit.arms)]
#                     reward_cols = ['reward_' + str(i + 1) for i in range(self.bandit.arms)]
#                     df = pd.DataFrame(data, columns=reward_cols + p_rew_cols)

#                     rin = 0  # Ugly quick fix reward instance = 1
#                     file_name = zip_name.replace('.zip', '') + '_rin_{}.csv'.format(str(rin)).lower()

#                     # Create CSV from dataframe
#                     df.to_csv(file_name)
#                     # Write CSV to zip file
#                     my_zip.write(file_name)
#                     # Delete CSV file
#                     os.remove(file_name)

#                 else:
#                     raise ValueError('This functionality is not implemented yet')

                    
                    
                
                
                    
                    
            
        

        
        

        
        
