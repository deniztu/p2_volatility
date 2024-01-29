# RNN test data 

Saved by default to `data/rnn_raw_data/`

The test dataframe is a pickled multiindex pandas.DataFrame with the following outer level indices: 
 - rnn_type:
     
     - 'lstm'
     - 'lstm2' (if noise == 'update-dependant')
     - 'rnn'
     
 - learning_algorithm:
 
     - 'a2c' (Advantage-Actor-Critic)
     - 'reinforce' (default)
 
 - noise:
     - 'none'
     - 'update-dependant' (Weber Noise, as in Findling & Wyart, 2020)
     
 - train_sd:
     The standard deviation of the gaussian noise distribution of the training bandit
 - rnn_id:
     The identifier of the RNN agent
 - test_sd:
     The standard deviation of the gaussian noise distribution of the test bandit
 - run:
     The bandit reward walk instance
 - reward_instance:
     Only relevant in binary rewards, identifier for a realisation in reward outcomes given the same reward probabilities
        
And following columns:
- bandit_parameter:
    - sd_noise: standard deviation of the gaussion noise (for restless bandits)
    - p_rew_best: probability of reward of the best action (for stationary bandits)
- choice:
    actions taken
- reward:
    rewards received
- value:
    estimated state value at each step
- p_rew_[i]:
    probability of reward for each action i at each step
- softmax_[i]:
    softmax probability of each action i at each step
- rnn_state_[i]:
    activity of hidden unit i at each step
- added_noise_rnn_state_[i]:
    noise added to hidden unit i at each step
- entropy_loss_weight:
    The weight of the entropy loss in the overall loss function. Default is 0.
- value_loss_weight:
    The weight of the value loss in the overall loss function. Default is 0.
- accuracy:
    binary indicator whether the agent chose the most rewarding action in a timestep
- is_switch:
    binary indicator whether the agent switched actions during subsequent timesteps

 
