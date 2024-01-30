# RNN test data 

Saved by default to `data/rnn_raw_data/`

The test dataframe is a pickled multiindex pandas.DataFrame with the following outer level indices: 

| Column name     |  Possible Values                                            |
|-----------------|-------------------------------------------------------------|
| rnn_type        | 'lstm' or 'lstm2' (if noise == 'update-dependant') or 'rnn' |
| learning_algorithm        | 'a2c' (Advantage-Actor-Critic) or 'reinforce' (default) |
| noise        | 'none' or 'update-dependant' (Weber Noise, as in Findling & Wyart, 2020) |
| train_sd        | The standard deviation of the gaussian noise distribution of the training bandit |
| rnn_id        | The identifier of the RNN agent |
| test_sd        | The standard deviation of the gaussian noise distribution of the test bandit |
| run        | The bandit reward walk instance |
| reward_instance        | Only relevant in binary rewards, identifier for a realisation in reward outcomes given the same reward probabilities |

        
And following columns:

| Column name     |  Possible Values                                            |
|-----------------|-------------------------------------------------------------|
| bandit_parameter        | standard deviation of the gaussion noise |
| choice        | actions taken |
| reward        | rewards received|
| value       | estimated state value at each stept |
| p_rew_[i]       | probability of reward for each action i at each step |
| softmax_[i]        |softmax probability of each action i at each stept |
| rnn_state_[i]        | activity of hidden unit i at each step |
| added_noise_rnn_state_[i]        | noise added to hidden unit i at each step |
| entropy_loss_weight        | weight of the entropy loss in the overall loss function. Default is 0. |
| value_loss_weight       | weight of the value loss in the overall loss function. Default is 0. |
| accuracy        |  binary indicator whether the agent chose the most rewarding action in a timestep |
| is_switch        | binary indicator whether the agent switched actions during subsequent timesteps |

# Preprocessed data for modeling

Saved by default to `data/intermediate_data/modeling/preprocessed_data_for_modeling/`

Preprocessed data is a .RData file containing a `res` object, with following 6 elements
- model: A string denoting the file name of the unprocessed file
- choices: A 1 by n_trials array containing the choices of the agent
- chosen_rewards: A 1 by n_trials array containing the rewards corresponding to choices
- rewards: A n_trials by n_actions array containing the reward values of all actions (random walks)
- nRuns: An integer indicating the number of subjects to fit in the stan model, can also be used to fit multiple runs for one agent
- nTrials: An integer indicating the number of trials of the task
 
