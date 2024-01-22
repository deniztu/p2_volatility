# -*- coding: utf-8 -*-

'''
The code is a Python script that trains and tests multiple instances of a recurrent neural network (RNN)
on different bandit problems, using parallel processing to speed up the computation time.
'''

# Import necessary modules
from classes.bandits import bandit_class as bc    # Load bandit class
from classes.neural_networks import network_class_organized_lstm_cell as nn    # Load neural network class
import concurrent.futures    # Load for parallel processing
import time    # Load for timing

# globals
N_HIDDEN = [11] # number of hidden units
ENTROPIES = ['linear'] # use 'linear' for linear decreasing entropy from 1 to 0
IDS = [199] # ids of the RNN instances to train and test

# Daw walks to test the RNNs on
daw_walks = ['classes/bandits/Daw2006_payoffs1.csv',
              'classes/bandits/Daw2006_payoffs2.csv',
              'classes/bandits/Daw2006_payoffs3.csv']

def tf_function(id_):
    
    '''Train and test RNN instances.
    
    id_: An integer representing the id of the RNN instance to train and test.
    '''
    
    # Define the number of hidden units
    for nh in N_HIDDEN:
    
        # Define the entropy values to use for training
        entropies = ENTROPIES
        
        # Loop over each entropy value
        for ent in entropies:
            
            # Define the bandit problem to use for training
            train_mab = bc.bandit(bandit_type='restless',
                                  arms=4,
                                  num_steps=300,
                                  reward_type='continuous',
                                  noise_sd=0.1,
                                  punish=True)
            
            # Define the RNN instance to train and test
            nnet = nn.neural_network(bandit=train_mab,
                                     noise='update-dependant',
                                     discount_rate=0.5,
                                     value_loss_weight=0.5,
                                     entropy_loss_weight=ent,
                                     rnn_type='lstm2',
                                     noise_parameter=0.5,
                                     learning_algorithm='a2c',
                                     n_iterations=500,
                                     model_id=id_,
                                     n_hidden_neurons=nh)
            
            # Train the RNN instance
            nnet.train()
            
            # Reset the RNN instance
            nnet.reset()
                    
            # Loop over each bandit problem to use for testing
            for daw_walk in range(3):
                
                # Get the bandit problem to use for testing
                test_mab = daw_walks[daw_walk]
                
                # Test the RNN instance on the bandit problem
                nnet.test(bandit=test_mab, bandit_param_range=[daw_walk+1], n_runs=1)

if __name__ == '__main__':
    
    # Define the ids of the RNN instances to train and test
    ids = IDS

    # Record the start time
    start = time.perf_counter()
    
    # Use parallel processing to train and test the RNN instances
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(tf_function, ids)
    
    # Record the finish time
    finish = time.perf_counter()
    
    # Print the total time taken to train and test the RNN instances
    print(f'Finished in {round(finish-start, 2)} second(s)')
