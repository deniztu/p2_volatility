# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:29:28 2022

@author: Deniz
"""

# load bandit class
from classes.bandits import bandit_class as bc
# load neural network class
from classes.neural_networks import network_class_organized_lstm_cell as nn
# load for parallel processing
import concurrent.futures
# load for timing
import time



def tf_function(id_):
    
    '''train and test RNN instances
    
    id_ 
    
    
    '''


    daw_walks = ['classes/bandits/Daw2006_payoffs1.csv',
                 'classes/bandits/Daw2006_payoffs2.csv',
                 'classes/bandits/Daw2006_payoffs3.csv']
    
    entropies = [0]
    
    
    for e in entropies:
        
        
   
    
        train_mab = bc.bandit(bandit_type = 'restless'
                            , arms = 4
                            , num_steps = 300
                            , reward_type = 'continuous'
                            , noise_sd = 0.1
                            , punish = True)
        
        nnet = nn.neural_network(bandit = train_mab
                            , noise = 'update-dependant'
                            , discount_rate = 0.5
                            , value_loss_weight= 0.5
                            , entropy_loss_weight = e
                            , rnn_type = 'lstm2'
                            , noise_parameter = 0.5
                            , learning_algorithm = 'a2c'
                            , n_iterations = 100000
                            , model_id= id_
                            , n_hidden_neurons = 48)
        
        # train the rnn
        nnet.train()
        
        # reset the rnn
        nnet.reset()
                
        for daw_walk in range(3):

            # daw walk
            test_mab = daw_walks[daw_walk]
            
            nnet.test(bandit = test_mab, bandit_param_range = [daw_walk+1], n_runs = 1)
            
if __name__ == '__main__':

    ids = range(14)

    start = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(tf_function, ids) # returns results in the order p's got started
    

    finish = time.perf_counter()
    
    print(f'Finished in {round(finish-start, 2)} second(s)')


        
        
