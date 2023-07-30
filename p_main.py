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

def tf_function(id_):
    
    '''Train and test RNN instances.
    
    id_: An integer representing the id of the RNN instance to train and test.
    '''
    
    # Define the number of hidden units
    for nh in [64, 80, 96]:
    
        # Define the entropy values to use for training
        entropies = [0]#[0, 0.05, 'linear']
        
        # Loop over each entropy value
        for e in entropies:
            
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
                                     entropy_loss_weight=e,
                                     rnn_type='lstm2',
                                     noise_parameter=0.5,
                                     learning_algorithm='a2c',
                                     n_iterations=50000,
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
    ids = range(16, 30)

    # Record the start time
    start = time.perf_counter()
    
    # Use parallel processing to train and test the RNN instances
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(tf_function, ids)    # Returns results in the order p's got started
    
    # Record the finish time
    finish = time.perf_counter()
    
    # Print the total time taken to train and test the RNN instances
    print(f'Finished in {round(finish-start, 2)} second(s)')
