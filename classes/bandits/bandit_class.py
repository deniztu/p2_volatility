import numpy as np
import random 
from scipy.stats import truncnorm as truncnorm

class Bandit:
    """Class to define a bandit
    
    Parameters
    ----------
    bandit_type : str
        Type of bandit, choose from 'stationary', 'restless', 'meta_volatility', 'daw_et_al_2006'.
    arms : int
        Number of bandit arms.
    num_steps : int
        Number of trials.
    dependant : bool
        Should reward probabilities of arms be dependent.
    reward_type : str
        Type of rewards, choose from 'continuous' or 'binary'(default).
    punish : bool
        If reward_type == 'binary': non-rewards = -1, if reward_type == 'continuous': rewards are mean-centered!
    
    For 'stationary' bandits:
        reward_rate : float
            Probability of observing reward.
    
    For 'restless' bandits:
        noise_sd : float
            Standard deviation of the Gaussian noise (mu = 0, sd = noise_sd). Default is 2.8.
    
    For 'meta_volatility' bandits:
        Restless bandits with noise_sd drawn from [0.02 - 0.2].
        
    Methods
    -------
    generate_task: Generates bandit task based on the specified bandit type and parameters
    
    Returns
    -------
    Bandit class instance
        
    Notes
    -----
    The method initializes a Bandit instance with the specified parameters.
    """
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
            
    def generate_task(self):
        
        """Generate bandit task based on the specified bandit type and parameters.
      
        Returns
        -------
        tuple
            Tuple containing rewards (numpy.ndarray) and reward probabilities (numpy.ndarray).
      
        Notes
        -----
        The method generates a bandit task based on the specified bandit type and parameters. It handles different
        scenarios such as restless and stationary bandits, with or without dependencies between arms.
      
        - For restless bandits:
            - If independent (`dependant` is False), it generates independant noisy reward probabilities for each arm over trials.
            - If dependent (`dependant` is True), it generates anti-correlated reward probabilities for each arm over trials.
      
        - For stationary bandits:
            - If independent (`dependant` is False), it randomly assigns the best bandit and generates rewards based
              on a fixed probability for the best arm and 0.5 for other arms.
            - If dependent (`dependant` is True), it generates dependent reward probabilities with the best arm having
              a fixed probability.
      
        - For meta-volatility bandits, it generates restless bandits with sampled standard deviation from uniform distribution U[0.02,0.2].
      
        - For Daw et al. 2006 bandits, it simulates a task with a decay parameter and diffusion noise (According to Daw et al., 2006).
      
        Returns a tuple containing rewards and reward probabilities.
      
        """
    
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
                
                rewards = r_probs
                
                if self.punish:
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
        
        else: 
            print('This functionality is not implemented yet')
        
