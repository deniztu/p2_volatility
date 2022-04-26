import numpy as np
import random 
import time
from scipy.stats import truncnorm as truncnorm
import pdb

class bandit:
    '''
    Class to define a bandit
    
    Input: 
        bandit_type: string, either 'stationary' or 'restless' or 'meta_volatility' or 'daw_et_al_2006'
        arms: int, number of bandit arms
        num_steps: int, number of trials
        dependant: bool, should reward probs of arms be dependant
        reward_type: 'continuous' or 'binary', note if 'continuous' rewards are mean centered!
        punish: bool, should non-rewards be negative

        if bandit_type == 'stationary':
            reward_rate: float, probability of observing reward
        
        if bandit_type == 'restless':
            noise_sd: float, sd of the gaussian noise (mu = 0, sd = noise_sd)
            
        if bandit_type == 'meta_volatility':
            restless bandits with noise_sd drawn from [0.02 - 0.2]
    
    Output: 
        rewards: numpy.ndarray, shape[num_steps, arms]
        p_rew: numpy.ndarray, probability of reward per arm
            
    '''
    def __init__(self
                 , bandit_type
                 , arms
                 , num_steps
                 , noise_sd = 2.8
                 , reward_rate = None
                 , dependant = False
                 , punish = True
                 , reward_type = 'binary'):
        

        self.bandit_type = bandit_type
        self.arms = arms
        self.num_steps = num_steps
        self.dependant = dependant
        self.punish = punish
        self.reward_type = reward_type
        
        if self.bandit_type == 'restless':
            self.bandit_parameter = noise_sd
        
        if self.bandit_type == 'stationary':
            self.bandit_parameter = reward_rate
            
        if self.bandit_type == 'meta_volatility':
            self.bandit_parameter = 'meta'
            
        if self.bandit_type == 'daw_et_al_2006':
            self.bandit_parameter = noise_sd
            
        if self.bandit_type == 'fixed_ratio':
            self.bandit_parameter = reward_rate
            
        if self.bandit_type == 'variable_ratio':
            self.bandit_parameter = reward_rate

            
    def generate_task(self):
    
        if self.bandit_type == 'restless' and self.dependant == False:
            
            # define mu and sd of gaussian noise
            mu = 0
            sd = self.bandit_parameter
            
            # initialise obtained reward prob array
            r_probs = np.zeros([self.num_steps, self.arms])
            
            # create noisy reward probabilities
            for t in range(self.num_steps):
                
                # first reward prob is 0.5 for all arms
                if t == 0:
                    # choose starting point at 0.5
                    r_probs[t,:] = 0.5
                    continue
                             
                # add gaussian noise to reward probs of each arm 
                noise = np.random.normal(mu, sd, self.arms)
                r_probs[t,:] = r_probs[t-1,:]+noise
                # bound rew probs between 0-1
                while any(r_probs[t,:] >=1) or any(r_probs[t,:] <=0):
                    noise = np.random.normal(mu, sd, self.arms)
                    r_probs[t,:] = r_probs[t-1,:]+noise
                    
                    
            # calculate rewards        

            rewards      = np.zeros([self.num_steps, self.arms])          
            
            if self.reward_type == 'binary':
                # perform bernoulli trials with reward probs
                random_numb  = np.random.rand(self.num_steps, self.arms)
                rewards = (r_probs > random_numb) * 1.
                
                if self.punish:
                    rewards = 2*rewards-1
            
            if self.reward_type == 'continuous':
                # mean center rewards
                rewards = r_probs - np.mean(r_probs)
            
            return rewards, r_probs
        
        if self.bandit_type == 'restless' and self.dependant == True:
            
            # define mu and sd of gaussian noise
            mu = 0
            sd = self.bandit_parameter
            
            # initialise obtained reward prob array for reference arm
            r_probs = np.zeros(self.num_steps)
            
            # create noisy reward probabilities
            for t in range(self.num_steps):
                
                # first reward prob is 0.5 for reference arm
                if t == 0:
                    # choose starting point at 0.5
                    r_probs[t] = 0.5
                    continue
                             
                # add gaussian noise to reward probs of reference arm 
                noise = np.random.normal(mu, sd, 1)
                r_probs[t] = r_probs[t-1]+noise
                # bound rew probs between 0-1
                while r_probs[t] >=1 or r_probs[t] <=0:
                    noise = np.random.normal(mu, sd, 1)
                    r_probs[t] = r_probs[t-1]+noise
                    
            # create dependant reference arm
            ref_arm     = np.random.randint(self.arms)
            
            # create dependant reward probability
            dep_rew_probs = np.zeros((self.num_steps, self.arms)) 
            for arm in range(self.arms):
                dep_rew_probs[:,arm] = 1- r_probs
            dep_rew_probs[:,ref_arm] = r_probs
            
            # calculate rewards        
            
            rewards      = np.zeros([self.num_steps, self.arms])       
            
            if self.reward_type == 'binary':
                # perform bernoulli trials with reward probs
                random_numb  = np.random.rand(self.num_steps, self.arms)
                rewards = (r_probs > random_numb) * 1.
                
                if self.punish:
                    rewards = 2*rewards-1
            
            if self.reward_type == 'continuous':
                # mean center rewards
                rewards = r_probs - np.mean(r_probs)
            
            return rewards, dep_rew_probs

        if self.bandit_type == 'meta_volatility' and self.dependant == False:
            # define mu and sd of gaussian noise
            mu = 0
            # choose sd randomly
            sd_vector = np.arange(0.02, 0.22, 0.02)
            sd = round(random.choice(sd_vector),2)
            
            # initialise obtained reward prob array
            r_probs = np.zeros([self.num_steps, self.arms])
            
            # create noisy reward probabilities
            for t in range(self.num_steps):
                
                # first reward prob is 0.5 for all arms
                if t == 0:
                    # choose starting point at 0.5
                    r_probs[t,:] = 0.5
                    continue
                             
                # add gaussian noise to reward probs of each arm 
                noise = np.random.normal(mu, sd, self.arms)
                r_probs[t,:] = r_probs[t-1,:]+noise
                # bound rew probs between 0-1
                while any(r_probs[t,:] >=1) or any(r_probs[t,:] <=0):
                    noise = np.random.normal(mu, sd, self.arms)
                    r_probs[t,:] = r_probs[t-1,:]+noise
                    
            
            # calculate rewards        
            
            rewards      = np.zeros([self.num_steps, self.arms])
            
            if self.reward_type == 'binary':
                # perform bernoulli trials with reward probs
                random_numb  = np.random.rand(self.num_steps, self.arms)
                rewards = (r_probs > random_numb) * 1.
                
                if self.punish:
                    rewards = 2*rewards-1
            
            if self.reward_type == 'continuous':
                # mean center rewards
                rewards = r_probs - np.mean(r_probs)
            

            
            return rewards, r_probs
        
        if self.bandit_type == 'stationary' and self.dependant == True:            
            
            # randomly assign best bandit
            rand_int     = np.random.randint(self.arms) 
            
            # initialise obtained reward prob array
            r_probs = np.zeros([self.num_steps, self.arms])
            
            # dependant arms (reward prob best arm 'p', all other 1-'p')
            r_probs[:]   = 1. - self.bandit_parameter
            r_probs[:, rand_int]   = self.bandit_parameter
            
            
            # calculate rewards        
            rewards      = np.zeros([self.num_steps, self.arms])
            
            if self.reward_type == 'binary':
                # perform bernoulli trials with reward probs
                random_numb  = np.random.rand(self.num_steps, self.arms)
                rewards = (r_probs > random_numb) * 1.
                
                if self.punish:
                    rewards = 2*rewards-1
            
            if self.reward_type == 'continuous':
                # mean center rewards
                rewards = r_probs - np.mean(r_probs)
                
            return rewards, r_probs
        
        if self.bandit_type == 'stationary' and self.dependant == False:            
            
            # randomly assign best bandit
            rand_int     = np.random.randint(self.arms) 
            
            # initialise obtained reward prob array
            r_probs = np.zeros([self.num_steps, self.arms])
            
            # dependant arms (reward prob best arm 'p', all other 1-'p')
            r_probs[:]   = .5
            r_probs[:, rand_int]   = self.bandit_parameter
            
            
            # calculate rewards        
            rewards      = np.zeros([self.num_steps, self.arms])
            
            if self.reward_type == 'binary':
                # perform bernoulli trials with reward probs
                random_numb  = np.random.rand(self.num_steps, self.arms)
                rewards = (r_probs > random_numb) * 1.
                
                if self.punish:
                    rewards = 2*rewards-1
            
            if self.reward_type == 'continuous':
                # mean center rewards
                rewards = r_probs - np.mean(r_probs)
                
            return rewards, r_probs
        
        if self.bandit_type == 'daw_et_al_2006':
            
            # decay parameter
            lambda_ = 0.9836
            # lambda_ = 0.9 # modify, because otherwise pay_off 'sticks' to bounds with higher sigma_d
            # decay center
            theta = 50
            # sigma (mu sd)
            sigma_0 = 4
            # sigma (diffusion sd)
            sigma_d = self.bandit_parameter
            
            # starting values for the means
            mu = np.random.uniform(25, 80, self.arms)
            
            # prepare payoff array
            pay_off_arr = np.zeros((self.num_steps, self.arms))
            
            # parameters for truncated normal
            myclip_a = 1
            myclip_b = 100

            for t in range(self.num_steps):
                
                # sample from a truncated normal(min=1, max=100)
                # define min, max according to documentation in scipy
                my_mean = mu
                my_std = sigma_0
                a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
                
                # sample
                pay_off = truncnorm.rvs(a, b, loc = my_mean, scale = my_std, size = self.arms)
                
                # round payoff to nearest integer (Daw et al. 2006)
                pay_off = np.round(pay_off)
                
                # append payoff
                pay_off_arr[t,:] = pay_off
                # diffusion noise
                noise = np.random.normal(loc = 0, scale = sigma_d, size = self.arms)
                # update mu 
                mu = lambda_*mu+(1-lambda_)*theta+noise
                
            # mean center rewards
            centered_pay_off_arr = pay_off_arr - np.mean(pay_off_arr)
                
            return(centered_pay_off_arr, pay_off_arr)
        
        if self.bandit_type == 'fixed_ratio':
            
            #pdb.set_trace()
            
            # randomly assign rewarded bandit
            # rand_int     = np.random.randint(self.arms)
            
            # initialise obtained reward prob array
            r_probs = np.zeros([self.num_steps, self.arms])
            
            # rewarded bandit is rewarded with fixed ratio, punished otherwise
            # unrewarded bandit has reward = 0 
            r_probs[:]   = 0
            r_probs[:, 1]   = - 1/8
            
            r_probs[::int(np.reciprocal(self.bandit_parameter)), 1] = 1
            
            # extinction phase
            r_probs[int(self.num_steps/2):, 1] = - 1/8
            
            rewards = r_probs
                
            return rewards, r_probs
        
        if self.bandit_type == 'variable_ratio':
            
            # randomly assign bandit for rewarded bandit
            # rand_int     = np.random.randint(self.arms)
            
            # initialise obtained reward prob array
            r_probs = np.zeros([self.num_steps, self.arms])
            
            # rewarded bandit is rewarded with fixed ratio, punished otherwise
            # unrewarded bandit has reward = 0 
            r_probs[:]   = 0
            r_probs[:, 1]   = - 1/8
            
            # select reward with variable ratio
            sample = r_probs[:int(self.num_steps/2), 1]
            sample[::int(np.reciprocal(self.bandit_parameter))] = 1
            r_probs[:int(self.num_steps/2), 1] = random.sample(list(sample),int(self.num_steps/2))
            
            # extinction phase all bandits not rewarded
            # r_probs[int(self.num_steps/2):] = 0
            
            rewards = r_probs
                
            return rewards, r_probs
        
        else: 
            print('This functionality is not implemented yet')
        
