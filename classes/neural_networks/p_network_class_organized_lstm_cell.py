import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy import signal
import numpy as np
import pandas as pd
import pickle
import glob
import os

from classes.neural_networks.rnns import recurrent_networks
from classes.neural_networks.rnns import own_lstm_cell
from classes.bandits import fixed_bandit_class as fbc
from classes.bandits import fixed_daw_bandit_class as fdbc
from helpers import dot2_
from helpers import zip2csv

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# class to generate the bandit tasks
class conditioning_bandit():
    def __init__(self, game):
        self.game = game        
        self.reset()
        
    def set_restless_prob(self):
        self.bandit         = self.restless_rewards[self.timestep]
        
    def reset(self):
        self.timestep          = 0 
        rewards , reward_probs = self.game.generate_task()     
        self.restless_rewards  = rewards
        self.reward_probs  = reward_probs
        self.set_restless_prob()
        
    def pullArm(self,action):
        if self.timestep >= (len(self.restless_rewards) - 1): done = True
        else: done = False
        return self.bandit[int(action)], done, self.timestep

    def update(self):
        self.timestep += 1
        self.set_restless_prob()
        
        
class AC_Network():
    def __init__(self, trainer, noise, rnn_type, noise_parameter,
                 n_hidden_neurons, n_arms, entropy_loss_weight,
                 value_loss_weight, learning_algorithm):    

        '''
        Returns the graph.
        Takes as input:
            trainer: a TensorFlow optimizer
            noise: with computation noise (noise=1) or decision entropy (noise=0)
            noise_parameter: coefficient for the computation noise or decision entropy
            rnn_type: the type of the recurrent neural network (LSTM or LSTM2)
            n_hidden_neurons: the number of neurons in the hidden layer
            n_arms: the number of actions available
            entropy_loss_weight: the weight given to the entropy loss
            value_loss_weight: the weight given to the value loss
            learning_algorithm: the name of the learning algorithm to use (e.g. REINFORCE)
        '''
        
        # Input
        self.prev_rewardsch = tf.placeholder(shape=[None,1], dtype=tf.float32, name="v1") # placeholder for previous rewards
        self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="v2") # placeholder for previous actions taken
        self.prev_actions_onehot = tf.one_hot(self.prev_actions, n_arms, dtype=tf.float32, name="v3") # one-hot encoding of previous actions
        self.timestep = tf.placeholder(shape=[None,1], dtype=tf.float32, name="v4") # placeholder for the current timestep
        input_ = tf.concat([self.prev_rewardsch, self.prev_actions_onehot], 1, name="v5") # concatenate previous rewards and actions into a single input tensor
        
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="v6") # placeholder for the current action
        self.actions_onehot = tf.one_hot(self.actions, n_arms, dtype=tf.float32, name="v7") # one-hot encoding of current action
        
        # set the entropy loss weight based on the input parameter
        if entropy_loss_weight == 'linear':
            self.entropy_loss_weight = tf.placeholder("float", None, name="v8") # if 'linear', create a placeholder for the weight
        else:
            self.entropy_loss_weight = entropy_loss_weight # otherwise, use the input value directly
        
        print('entropy loss weight')
        print(self.entropy_loss_weight) # print the entropy loss weight (either the input value or the placeholder)
        
        # Set the number of units in the hidden layer
        nb_units = n_hidden_neurons
        
        # If the RNN type is LSTM and there is no noise
        if rnn_type == 'lstm' and noise == 'none':
        
            # Create a basic LSTM cell with nb_units number of units
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(nb_units, state_is_tuple=True)
            
            # Set the initial state of the cell to zeros
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            
            # Create placeholders for the cell's state
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            
            # Expand the dimensions of the input to be [batch_size, max_time, input_dim]
            # The batch_size is 1, max_time is determined by the length of the sequence (step_size), and input_dim is the number of features in the input
            rnn_in = tf.expand_dims(input_, [0])
            
            # Create a tensor to represent the length of the sequence
            step_size = tf.shape(self.prev_rewardsch)[:1]
            
            # Create an LSTM state tuple from the input cell state
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            
            # Run the LSTM cell on the input sequence, starting from the initial state
            # Outputs will be of size [batch_size, max_time, nb_units]
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size, time_major=False)
            
            # Get the final cell and hidden states
            lstm_c, lstm_h = lstm_state
            
            # Create a tensor to hold the means of added noises for the REINFORCE algorithm
            self.added_noises_means = tf.convert_to_tensor(np.zeros(48))
            
            # Create a placeholder for the noise to be added to the hidden state (for training with REINFORCE algorithm)
            self.h_noise = tf.placeholder(tf.float32, [None, nb_units])
            
            # Set the output state to be the final cell and hidden states
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            
            # Set the true output state to be the final hidden state
            self.true_state_out = lstm_h[:1, :]
            
            # Reshape the outputs to be [batch_size * max_time, nb_units]
            rnn_out = tf.reshape(lstm_outputs, [-1, nb_units])
            
        # Check if the LSTM cell type is 'lstm2' and noise type is 'update-dependant'
        if rnn_type == 'lstm2' and noise == 'update-dependant':
        
            # Set flag to add noise to LSTM cell
            add_noise = True
        
            # Define basic LSTM cell with the given number of units (neurons)
            lstm_cell_b = tf.contrib.rnn.BasicLSTMCell(nb_units,state_is_tuple=True)
        
            # Define custom LSTM cell with the given number of arms and units, and whether to add noise
            lstm_cell = own_lstm_cell.LSTM(n_arms+1, nb_units, add_noise)
        
            # Set initial values for the LSTM cell state
            c_init = np.zeros((1, nb_units), np.float32)
            h_init = np.zeros((1, nb_units), np.float32)
            self.state_init = [h_init, c_init]
        
            # Define placeholder tensors for the current state of the LSTM cell
            c_in = tf.placeholder(tf.float32, [1, nb_units])
            h_in = tf.placeholder(tf.float32, [1, nb_units])
            self.state_in = tf.tuple([h_in, c_in])
        
            # Define placeholder tensor for the noise to be added to the LSTM cell
            self.h_noise = tf.placeholder(tf.float32, [None, nb_units])
            all_noises = self.h_noise
        
            # Concatenate the input and noise tensors along the feature (i.e. neuron) axis
            all_inputs = tf.concat((input_, all_noises), axis=1)
            rnn_in = tf.transpose(tf.expand_dims(all_inputs, [0]), [1, 0, 2])
        
            # Pass the concatenated tensor through the LSTM cell using the scan function
            # Initial state is set to the current state of the cell
            states = tf.scan(lstm_cell.step, rnn_in, initializer=(self.state_in))
        
            # Define a tensor to hold the added noise means for use in the reinforce algorithm
            self.added_noises_means = tf.convert_to_tensor(np.zeros(48))
        
            # Extract the output state tensors from the states tensor
            lstm_h, lstm_c = states
        
            # Set the output state of the LSTM cell to be the first time step of the output tensors
            self.state_out = (lstm_h[0,:, :], lstm_c[0,:, :]) 
            
            # Extract the first time step of the output tensor to use as the "true" output state
            self.true_state_out = lstm_h[:1, :]
        
            # Reshape the output tensor to have the same number of neurons as the input tensor
            rnn_out = tf.reshape(lstm_h, [-1, nb_units])

        if rnn_type == 'rnn':
            # Check the noise type to determine whether or not to add noise
            if noise == 'update-dependant':
                add_noise = True
            if noise == 'none':
                add_noise = False
            if noise == 'constant':
                raise ValueError('Constant Noise in RNN not implemented yet!')
            
            # Create the RNN cell
            lstm_cell = recurrent_networks.RNN(n_arms+1, nb_units, add_noise) # input (last reward, one-hot actions)
            
            # Set the initial state to zeros
            h_init = np.zeros((1, nb_units), np.float32)
            self.state_init = [h_init]
            
            # Define the input and noise placeholders
            self.h_in = tf.placeholder(tf.float32, [1, nb_units])
            self.h_noise = tf.placeholder(tf.float32, [None, nb_units])
            
            # Set the initial state to h_in
            self.state_in = self.h_in
            
            # Concatenate the input and noise if add_noise is True
            all_noises = self.h_noise
            if add_noise: 
                all_inputs = tf.concat((input_, all_noises), axis=1)
                rnn_in = tf.transpose(tf.expand_dims(all_inputs, [0]),[1,0,2])
            else:
                rnn_in = tf.transpose(tf.expand_dims(input_, [0]),[1,0,2])
            
            # Perform the RNN operation with the scan function
            states, self.added_noises_means = tf.scan(lstm_cell.step, rnn_in, initializer=(self.state_in, 0.))
            
            # Extract the hidden state from the RNN output
            lstm_h = states[:,0]
            
            # Set the state output to the first element of the RNN output
            self.state_out = states[:1,0]
            
            # Set the true state output to the state output
            self.true_state_out = self.state_out
            
            # Set the RNN output to the hidden state
            rnn_out = lstm_h

        # Loss functions
        
        # Softmax layer for generating action probabilities from the LSTM output
        self.policy = slim.fully_connected(rnn_out, n_arms, activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
        
        # Placeholder for advantages
        self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
        
        # Calculate the probability of taking the selected action
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
        
        # Calculate the entropy of the output distribution to encourage exploration
        self.entropy     = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
        
        # Calculate the policy loss, which is the negative log-likelihood of the chosen action multiplied by the advantage
        self.policy_loss = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.advantages)
        
        # Calculate the entropy loss, which encourages exploration by penalizing low-entropy output distributions
        self.loss_entropy = self.entropy * self.entropy_loss_weight
        
        # If using A2C algorithm, calculate the value loss as well
        if learning_algorithm == 'a2c':
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
            self.loss        = value_loss_weight * self.value_loss + self.policy_loss - self.loss_entropy
        
        # If using REINFORCE algorithm, only calculate policy loss and entropy loss
        if learning_algorithm == 'reinforce':
            self.loss        = self.policy_loss - self.loss_entropy
        
        # Get gradients from network using losses
        local_vars            = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients        = tf.gradients(self.loss,local_vars)
        
        # Calculate the gradient norm (added)
        self.gradient_norm    = tf.global_norm(self.gradients)
        
        self.var_norms        = tf.global_norm(local_vars)         
        
        # Apply gradients to update network weights
        self.apply_grads      = trainer.apply_gradients(zip(self.gradients,local_vars))
        
class Worker():
    def __init__(self, game, trainer, model_path, model_name, noise
                 , path_to_save_progress, n_hidden_neurons, n_arms, num_steps
                 , n_iterations, rnn_type, noise_parameter, entropy_loss_weight
                 , value_loss_weight, learning_algorithm):
        
        self.model_path            = model_path
        self.trainer               = trainer
        self.episode_rewards       = []
        self.episode_lengths       = []
        self.addnoises_mean_values = []
        self.hidden_mean_values    = []
        self.episode_reward_reversal = []
        self.summary_writer        = tf.summary.FileWriter(path_to_save_progress + str(model_name))
        
        self.ac_network = AC_Network(trainer = trainer, noise = noise, rnn_type = rnn_type
                                     , noise_parameter = noise_parameter
                                     , n_hidden_neurons = n_hidden_neurons, n_arms = n_arms
                                     , entropy_loss_weight = entropy_loss_weight
                                     , value_loss_weight = value_loss_weight
                                     , learning_algorithm = learning_algorithm)
        self.env      = game
        self.num_steps = num_steps
        self.n_iterations = n_iterations
        self.n_hidden_neurons = n_hidden_neurons
        self.n_arms = n_arms
        self.learning_algorithm = learning_algorithm
        self.rnn_type = rnn_type
        self.noise_parameter = noise_parameter
        self.noise = noise
        self.entropy_loss_weight = entropy_loss_weight

    def train(self, rollout, sess, gamma, bootstrap_value, entr_):
        '''
        Trains the network.
        Args:
            rollout (list): Rollout trajectory
            sess (tf.Session): TensorFlow session
            gamma (float): Discount factor
            bootstrap_value (float): The value of the last state in the rollout
            entr_ (float): The entropy regularization coefficient
        Returns:
            value_loss (float): The value loss
            policy_loss (float): The policy loss
            entropy (float): The entropy regularization term
            v_n (float): The value function's L2 norm
            grad_ (float): The policy's gradient L2 norm
        '''

        # Convert the rollout to a numpy array
        rollout = np.array(rollout)
        
        # Extract the actions, rewards, timesteps, and noise values from the rollout
        actions = rollout[:, 0]
        rewards_ch = rollout[:, 1]
        timesteps = rollout[:, 2]
        h_noises = rollout[:, 3]
        
        # Initialize the previous action and reward values
        prev_actions = [2] + actions[:-1].tolist()
        prev_rewards_ch = [0] + rewards_ch[:-1].tolist()
        
        # If using A2C, calculate the advantages
        if self.learning_algorithm == 'a2c':
            values = rollout[:, 4]
            self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
            advantages = rewards_ch + gamma * self.value_plus[1:] - self.value_plus[:-1]
            advantages = discount(advantages, gamma)
            
        # If using REINFORCE, initialize the values tensor
        if self.learning_algorithm == 'reinforce':
            values = tf.convert_to_tensor(np.zeros(300))
            
        # Calculate the discounted rewards
        self.rewards_plus = np.asarray(rewards_ch.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        
        if self.rnn_type == 'lstm' and self.learning_algorithm == 'a2c' or self.rnn_type == 'lstm2' and self.learning_algorithm == 'a2c':
        # Check if using LSTM and A2C algorithm, or LSTM2 and A2C algorithm
        
            # Initialize the RNN state for the current rollout
            rnn_state = self.ac_network.state_init 
            
            if self.entropy_loss_weight == 'linear':
                # If using linear entropy loss weight, set up the feed_dict with all necessary inputs
                
                feed_dict = {
                    self.ac_network.target_v: discounted_rewards,  # discounted reward
                    self.ac_network.prev_rewardsch: np.vstack(prev_rewards_ch),  # previous rewards
                    self.ac_network.prev_actions: prev_actions,  # previous actions
                    self.ac_network.h_noise: np.vstack(h_noises),  # noise added to LSTM hidden state
                    self.ac_network.actions: actions,  # current actions
                    self.ac_network.timestep: np.vstack(timesteps),  # time steps
                    self.ac_network.advantages: advantages,  # advantage estimate
                    self.ac_network.state_in[0]: rnn_state[0],  # LSTM hidden state
                    self.ac_network.state_in[1]: rnn_state[1],  # LSTM cell state
                    self.ac_network.entropy_loss_weight: entr_  # entropy loss weight
                }
                
            else:
                # If not using linear entropy loss weight, set up the feed_dict without it
                
                feed_dict = {
                    self.ac_network.target_v: discounted_rewards, 
                    self.ac_network.prev_rewardsch: np.vstack(prev_rewards_ch),
                    self.ac_network.prev_actions: prev_actions,
                    self.ac_network.h_noise: np.vstack(h_noises),
                    self.ac_network.actions: actions,
                    self.ac_network.timestep: np.vstack(timesteps),
                    self.ac_network.advantages: advantages,
                    self.ac_network.state_in[0]: rnn_state[0],
                    self.ac_network.state_in[1]: rnn_state[1]
                }
            
            # Run the TensorFlow session to compute the value loss, policy loss, entropy, and gradient norm
            # Then, update the network weights
            v_l, p_l, e_l, v_n, _, grad_ = sess.run([self.ac_network.value_loss,
                                                      self.ac_network.policy_loss,
                                                      self.ac_network.entropy,
                                                      self.ac_network.var_norms,
                                                      self.ac_network.apply_grads, 
                                                      self.ac_network.gradient_norm],
                                                     feed_dict=feed_dict)            
    
            return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), 0., v_n, grad_ 
            # Return the losses, variance norm, and gradient norm, averaged over the rollout
        
        # Check if the specified recurrent neural network type is 'lstm2' and the learning algorithm is 'reinforce'
        if self.rnn_type == 'lstm2' and self.learning_algorithm == 'reinforce':
                    
            # Initialize the state of the recurrent neural network
            rnn_state = self.ac_network.state_init ###change
                    
            # Check if the entropy loss weight is set to 'linear'
            if self.entropy_loss_weight == 'linear':
                        
                # Create a feed dictionary for the session to compute the values of the specified tensors
                feed_dict = {self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                             self.ac_network.prev_actions:prev_actions,
                             self.ac_network.h_noise:np.vstack(h_noises), 
                             self.ac_network.actions:actions,
                             self.ac_network.timestep:np.vstack(timesteps),
                             self.ac_network.advantages:discounted_rewards,
                             self.ac_network.state_in[0]:rnn_state[0],
                             self.ac_network.state_in[1]:rnn_state[1],
                             self.ac_network.entropy_loss_weight: entr_}   
                        
            else:
                # Create a feed dictionary without the entropy loss weight parameter for the session
                feed_dict = {self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                             self.ac_network.prev_actions:prev_actions,
                             self.ac_network.h_noise:np.vstack(h_noises),
                             self.ac_network.actions:actions,
                             self.ac_network.timestep:np.vstack(timesteps),
                             self.ac_network.advantages:discounted_rewards,
                             self.ac_network.state_in[0]:rnn_state[0],
                             self.ac_network.state_in[1]:rnn_state[1]} 
                        
            # Run the session with the specified tensors and feed dictionary
            p_l,e_l,v_n,_, grad_ = sess.run([self.ac_network.policy_loss,
                                              self.ac_network.entropy,
                                              self.ac_network.var_norms,
                                              self.ac_network.apply_grads, 
                                              self.ac_network.gradient_norm],
                                              feed_dict=feed_dict)
            
            # Return the policy loss, entropy loss, a scalar value of 0, variance norms, and gradient norms 
            # divided by the length of the rollout trajectory 
            return p_l / len(rollout),e_l / len(rollout), 0.,v_n, grad_ 
        
        # Check if the RNN type is 'lstm' and the learning algorithm is 'reinforce'
        if self.rnn_type == 'lstm' and self.learning_algorithm == 'reinforce':
            
            # Set the RNN state to the initial state
            rnn_state = self.ac_network.state_init
            
            # Check if the entropy loss weight is set to 'linear'
            if self.entropy_loss_weight == 'linear':
                
                # Define a feed dictionary with various network inputs and the entropy loss weight
                feed_dict = {
                    self.ac_network.prev_rewardsch: np.vstack(prev_rewards_ch),
                    self.ac_network.prev_actions: prev_actions,
                    self.ac_network.h_noise: np.vstack(h_noises), 
                    self.ac_network.actions: actions,
                    self.ac_network.timestep: np.vstack(timesteps),
                    self.ac_network.advantages: discounted_rewards,
                    self.ac_network.state_in[0]: rnn_state[0],
                    self.ac_network.state_in[1]: rnn_state[1],
                    self.ac_network.entropy_loss_weight: entr_
                }   
                
            # If the entropy loss weight is not set to 'linear'
            else:
                
                # Define a feed dictionary with various network inputs
                feed_dict = {
                    self.ac_network.prev_rewardsch: np.vstack(prev_rewards_ch),
                    self.ac_network.prev_actions: prev_actions,
                    self.ac_network.h_noise: np.vstack(h_noises),
                    self.ac_network.actions: actions,
                    self.ac_network.timestep: np.vstack(timesteps),
                    self.ac_network.advantages: discounted_rewards,
                    self.ac_network.state_in[0]: rnn_state[0],
                    self.ac_network.state_in[1]: rnn_state[1]
                } 
                
            # Run the session to calculate the policy loss, entropy, variable norms, and gradients
            p_l, e_l, v_n, _, grad_ = sess.run([
                self.ac_network.policy_loss,
                self.ac_network.entropy,
                self.ac_network.var_norms,
                self.ac_network.apply_grads, 
                self.ac_network.gradient_norm
            ], feed_dict=feed_dict)
            
            # Return the policy loss divided by the length of the rollout, entropy loss divided by length of the rollout,
            # 0, variable norms, and gradients
            return p_l / len(rollout), e_l / len(rollout), 0., v_n, grad_
        
        if self.rnn_type == 'rnn' and self.learning_algorithm == 'a2c':
            
            rnn_state = self.ac_network.state_init[0]
            
            if self.entropy_loss_weight == 'linear':
                                
                feed_dict = {self.ac_network.target_v:discounted_rewards,
                             self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                             self.ac_network.prev_actions:prev_actions,
                             self.ac_network.h_noise:np.vstack(h_noises),
                             self.ac_network.actions:actions,
                             self.ac_network.timestep:np.vstack(timesteps),
                             self.ac_network.advantages:advantages,
                             self.ac_network.h_in:rnn_state,
                             self.ac_network.entropy_loss_weight: entr_}   
                
            else:
                feed_dict = {self.ac_network.target_v:discounted_rewards,
                             self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                             self.ac_network.prev_actions:prev_actions,
                             self.ac_network.h_noise:np.vstack(h_noises),
                             self.ac_network.actions:actions,
                             self.ac_network.timestep:np.vstack(timesteps),
                             self.ac_network.advantages:advantages,
                             self.ac_network.h_in:rnn_state}
            
            v_l, p_l,e_l,v_n,_, grad_ = sess.run([self.ac_network.value_loss,
                                      self.ac_network.policy_loss,
                                      self.ac_network.entropy,
                                      self.ac_network.var_norms,
                                      self.ac_network.apply_grads, 
                                      self.ac_network.gradient_norm],
                                      feed_dict=feed_dict)            

            return v_l / len(rollout), p_l / len(rollout),e_l / len(rollout), 0.,v_n, grad_ 
        
        if self.rnn_type == 'rnn' and self.learning_algorithm == 'reinforce':
                        
            rnn_state = self.ac_network.state_init[0]
            
            if self.entropy_loss_weight == 'linear':
                                
                feed_dict = {self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                             self.ac_network.prev_actions:prev_actions,
                             self.ac_network.h_noise:np.vstack(h_noises),
                             self.ac_network.actions:actions,
                             self.ac_network.timestep:np.vstack(timesteps),
                             self.ac_network.advantages:discounted_rewards,
                             self.ac_network.h_in:rnn_state,
                             self.ac_network.entropy_loss_weight: entr_}
            else:
            
                feed_dict = {self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                             self.ac_network.prev_actions:prev_actions,
                             self.ac_network.h_noise:np.vstack(h_noises),                     
                             self.ac_network.actions:actions,
                             self.ac_network.timestep:np.vstack(timesteps),
                             self.ac_network.advantages:discounted_rewards,
                             self.ac_network.h_in:rnn_state}            

            p_l,e_l,v_n,_, grad_ = sess.run([self.ac_network.policy_loss,
                                             self.ac_network.entropy,
                                             self.ac_network.var_norms,
                                             self.ac_network.apply_grads, 
                                             self.ac_network.gradient_norm],
                                             feed_dict=feed_dict)
            
            return p_l / len(rollout),e_l / len(rollout), 0.,v_n, grad_
        
    def work(self, gamma, sess, saver, train):
        '''
        This is the main function
        Takes as input: gamma, the discount factor
                        sess, a Tensorflow session
                        saver, a Tensorflow saver
                        train boolean, do we train or not?
        The function will train the agent on the A task. To do so, the agent plays an A episode, and at the end of the episode, 
        we use the experience to perform a gradient update. When computation noise is assumed in the RNN, the noise realizations are 
        saved in the buffer and then fed to the back-propagation process.
        '''

        #################################################################################
        #################### if network is tested create a dataframe ####################
        #################################################################################
        
        if train == False:
            
            # prepare column names for probability of reward, softmax, unit activity and unit noise
            rnn_state_col_names = ['']*self.n_hidden_neurons
            rnn_state_noise_col_names = ['']*self.n_hidden_neurons
            
            rnn_prob_rew = ['']*self.n_arms
            rnn_softmax = ['']*self.n_arms
            
            # create column names for each unit of the RNN state and noise
            for i in range(self.n_hidden_neurons):
                rnn_state_col_names[i] = 'rnn_state_'+str(i+1)
                rnn_state_noise_col_names[i] = 'added_noise_rnn_state_'+str(i+1)
                
            # create column names for probability of reward and softmax for each action
            for i in range(self.n_arms):
                rnn_prob_rew[i] = 'p_rew_'+str(i+1)
                rnn_softmax[i] = 'softmax_'+str(i+1)           
            
            # prepare column name for the bandit parameter
            if self.env.game.bandit_type == 'restless':
                bandit_par = 'sd_noise'
            elif self.env.game.bandit_type == 'stationary':
                bandit_par = 'p_rew_best'
            else: 
                bandit_par = 'bandit parameter'
                
            # combine all column names
            colnames = [bandit_par,'choice', 'reward', 'value']
            colnames.extend(rnn_prob_rew + rnn_softmax + rnn_state_col_names + rnn_state_noise_col_names)
            
            # create an empty DataFrame with the specified columns
            df = pd.DataFrame(columns=colnames)
            
            # prepare variables to collect data
            my_a = [] # actions taken
            my_rch = [] # rewards received
            my_r_prob = np.zeros([self.num_steps, self.n_arms]) # probability of reward for each action at each step
            rnn_softmax_arr = np.zeros([self.num_steps, self.n_arms]) # softmax of probability of reward for each action at each step
            rnn_state_arr = np.zeros([ self.num_steps, self.n_hidden_neurons]) # RNN state at each step
            rnn_value_arr = np.zeros([self.num_steps]) # estimated value of current state at each step
            rnn_state_noise_arr = np.zeros([self.num_steps, self.n_hidden_neurons]) # noise added to RNN state at each step
            
        ##################################################################################
            
        # take episode count if model was already trained
        if os.path.exists(self.model_path):  # check if model directory exists
            # get list of files in the model directory and ignore the checkpoint file
            list_of_files = glob.glob(self.model_path + '/*')
            list_of_files = np.array(list_of_files)[[not "checkpoint" in s for s in list_of_files]]
            # find the latest file in the list based on creation time
            latest_file = max(list_of_files, key=os.path.getctime)
            # extract the episode count from the file name
            x = latest_file
            episode_count = int(x.split('-')[1].split('.')[0])
        else:
            episode_count = 0  # if model directory doesn't exist, set episode count to 0
        
        entr_ = 1  # set entr_ to 1
        
        while True:  # start an infinite loop for running episodes
            episode_buffer, state_mean_arr, added_noise_arr = [], [], []
            episode_reward, episode_step_count = 0, 0
            d, a, t, rch = False, 2, 0, 0  # set initialization parameters (d is a flag, a is a one-hot vector, t is the time step, and rch is the previous reward)
        
            # initialize the RNN state based on the RNN type
            if self.rnn_type == 'lstm':
                rnn_state = self.ac_network.state_init
            if self.rnn_type == 'lstm2':
                rnn_state = self.ac_network.state_init
            if self.rnn_type == 'rnn':
                rnn_state = self.ac_network.state_init[0]
        
            self.env.reset()  # reset the environment
        
            while d == False:  # start an episode loop until the episode is done
                # set the noise parameter based on the noise type
                if self.noise == 'update-dependant':
                    if self.rnn_type == 'lstm2':
                        h_noise = np.array(np.random.normal(size=rnn_state[0].shape) * self.noise_parameter, dtype=np.float32)
                    else:
                        h_noise = np.array(np.random.normal(size=rnn_state.shape) * self.noise_parameter, dtype=np.float32)
                if self.noise == 'constant':
                    raise ValueError('Constant noise not implemented yet!')
                if self.noise == 'none':
                    h_noise = np.array(np.random.normal(size=self.ac_network.state_init[0].shape) * self.noise_parameter, dtype=np.float32)

                
                # Check if the RNN type is LSTM
                if self.rnn_type == 'lstm':
                    # If the entropy loss weight is linear, set the feed_dict accordingly
                    if self.entropy_loss_weight == 'linear':
                        feed_dict = {
                            self.ac_network.prev_rewardsch:[[rch]],
                            self.ac_network.prev_actions:[a],
                            self.ac_network.timestep:[[t]],
                            self.ac_network.state_in[0]:rnn_state[0],
                            self.ac_network.state_in[1]:rnn_state[1],
                            self.ac_network.h_noise:h_noise, 
                            self.ac_network.entropy_loss_weight:entr_
                        }
                    # If the entropy loss weight is not linear, set the feed_dict accordingly
                    else:
                        feed_dict = {
                            self.ac_network.prev_rewardsch:[[rch]],
                            self.ac_network.prev_actions:[a],
                            self.ac_network.timestep:[[t]],
                            self.ac_network.state_in[0]:rnn_state[0],
                            self.ac_network.state_in[1]:rnn_state[1],
                            self.ac_network.h_noise:h_noise
                        }
                # If the RNN type is not LSTM but is LSTM2 instead, set the feed_dict accordingly
                elif self.rnn_type == 'lstm2':
                    if self.entropy_loss_weight == 'linear':
                        feed_dict = {
                            self.ac_network.prev_rewardsch:[[rch]],
                            self.ac_network.prev_actions:[a],
                            self.ac_network.timestep:[[t]],
                            self.ac_network.state_in[0]:rnn_state[0],
                            self.ac_network.state_in[1]:rnn_state[1],
                            self.ac_network.h_noise:h_noise, 
                            self.ac_network.entropy_loss_weight:entr_
                        }
                    else:
                        feed_dict = {
                            self.ac_network.prev_rewardsch:[[rch]],
                            self.ac_network.prev_actions:[a],
                            self.ac_network.timestep:[[t]],
                            self.ac_network.state_in[0]:rnn_state[0],
                            self.ac_network.state_in[1]:rnn_state[1],
                            self.ac_network.h_noise:h_noise
                        }
                        
                # If the RNN type is RNN, set the feed_dict accordingly
                elif self.rnn_type == 'rnn':
                    # Check the type of entropy loss weight
                    if self.entropy_loss_weight == 'linear':
                        
                        # Define the feed_dict with the current states and actions and entropy loss weight
                        feed_dict = {self.ac_network.prev_rewardsch:[[rch]],
                                     self.ac_network.prev_actions:[a],
                                     self.ac_network.timestep:[[t]],
                                     self.ac_network.h_in:rnn_state,
                                     self.ac_network.h_noise:h_noise, 
                                     self.ac_network.entropy_loss_weight:entr_}
                        
                    else:
                        
                        # Define the feed_dict with the current states and actions
                        feed_dict = {self.ac_network.prev_rewardsch:[[rch]],
                                     self.ac_network.prev_actions:[a],
                                     self.ac_network.timestep:[[t]],
                                     self.ac_network.h_in:rnn_state,
                                     self.ac_network.h_noise:h_noise}
                    
                # If the learning algorithm is A2C
                if self.learning_algorithm == 'a2c':
                    
                    # Run the policy network, value network, and the current and true state of the recurrent neural network (RNN)
                    # using the feed_dict to populate the placeholders with values of feed_dict.
                    a_dist, v, rnn_state_new, rnn_true_state_new = sess.run([self.ac_network.policy,
                                                                             self.ac_network.value,
                                                                             self.ac_network.state_out,
                                                                             self.ac_network.true_state_out],
                                                                            feed_dict=feed_dict)
                # If the learning algorithm is Reinforce
                if self.learning_algorithm == 'reinforce':
                    
                    # Run the policy network, current and true state of the recurrent neural network (RNN), and added noise values 
                    # using the feed_dict to populate the placeholders with values of feed_dict.
                    a_dist, rnn_state_new, rnn_true_state_new, added_noise = sess.run([self.ac_network.policy,
                                                                                       self.ac_network.state_out,
                                                                                       self.ac_network.true_state_out,
                                                                                       self.ac_network.added_noises_means],
                                                                                      feed_dict=feed_dict)
                
                    # Set value to 0
                    v = np.array([[0]])
                
                # Choose an action randomly from the probability distribution a_dist
                a = np.random.choice(a_dist[0], p=a_dist[0])
                
                # Convert the action to one-hot vector
                a = np.argmax(a_dist == a)
                
                # If the RNN type is LSTM
                if self.rnn_type == 'lstm':
                    
                    # Set rnn_state to rnn_state_new
                    rnn_state = rnn_state_new
                
                # If the RNN type is LSTM2
                if self.rnn_type == 'lstm2':
                    
                    # Set rnn_state to rnn_state_new
                    rnn_state = rnn_state_new
                
                # If the RNN type is RNN
                if self.rnn_type == 'rnn':
                    
                    # Set rnn_state to first two elements of rnn_state_new
                    rnn_state = rnn_state_new[:2]
                
                # Get the reward, done flag, and time step of the arm pulled
                rch, d, t = self.env.pullArm(a)
                
                # Add the reward to the episode_reward
                episode_reward += rch
                
                # Increase the episode_step_count by 1
                episode_step_count += 1 
                
                # Append the current action, reward, done flag, added noise, value, and rnn_state to episode_buffer
                episode_buffer.append([a, rch, t, h_noise, v[0,0], d])
                
                # If network is tested, collect variables
                if train == False: 
                    
                    # Append the action to my_a list
                    my_a.append(a)
                    
                    # Append the reward to my_rch list
                    my_rch.append(rch)
                    
                    # Set the reward probability at the current time step to the corresponding value in the environment
                    my_r_prob[t] = self.env.reward_probs[t]
                    
                    # Set rnn_softmax_arr at the current time step to the current probability distribution of actions
                    rnn_softmax_arr[t] = a_dist
                    
                    # Set rnn_value_arr at the current time step to the current value of the value network
                    rnn_value_arr[t]
                
                # if episode not done go to next trial
                if not d:
                    self.env.update()
                
            # collect for tensorflow
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_step_count)

            # Update the network using the experience buffer at the end of the episode.
            if len(episode_buffer) != 0 and train == True:
                
                if self.learning_algorithm == 'a2c':
                    
                    v_l, p_l,e_l,g_n,v_n, gg = self.train(episode_buffer,sess,gamma,0.0, entr_)
                    
                if self.learning_algorithm == 'reinforce':
                    
                    p_l,e_l,g_n,v_n, gg = self.train(episode_buffer,sess,gamma,0.0, entr_)

            # stop after first episode if model is tested
            if train == False:
                
                # populate dataframe 
                df['choice'] = my_a
                df['reward'] = my_rch
                df['value'] = rnn_value_arr
                df[rnn_prob_rew] = my_r_prob
                df[rnn_state_col_names] = rnn_state_arr
                df[rnn_state_noise_col_names] = rnn_state_noise_arr
                df[rnn_softmax] = rnn_softmax_arr
                df[bandit_par] = self.env.game.bandit_parameter
                
                return df

            # Periodically save summary statistics.
            if episode_count != 0:
                if episode_count % 500 == 0 and train == True:
                    
                    # create folder to save models
                    if not os.path.exists(self.model_path):
                        os.makedirs(self.model_path)
                    
                    saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.cptk')
                    print("Saved Model Episodes: {}".format(str(episode_count)))
                    mean_reward    = np.mean(self.episode_rewards[-50:])
                    print('mean_reward')
                    print(mean_reward)
                
                if train == True:
                    if episode_count % self.n_iterations == 0: # stopping criterion
                        return None

                mean_reward    = np.mean(self.episode_rewards[-50:])
                mean_noiseadd  = np.mean(self.addnoises_mean_values[-50:])
                mean_hidden    = np.mean(self.hidden_mean_values[-50:])
                # mean_reversal  = np.mean(self.episode_reward_reversal[-1])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                # summary.value.add(tag='Perf/reversal_Reward', simple_value=float(mean_reversal))
                summary.value.add(tag='Info/Noise_added', simple_value=float(mean_noiseadd))
                summary.value.add(tag='Info/Hidden_activity', simple_value=float(mean_hidden))
                # summary.value.add(tag='Parameters/biases_transition', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[3])).mean())
                summary.value.add(tag='Parameters/matrix_transition', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[1])).mean())                
                summary.value.add(tag='Parameters/matrix_input', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2])).mean())                                
                if train == True:
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    # added gg
                    summary.value.add(tag='Gradnorm', simple_value=float(gg))
                self.summary_writer.add_summary(summary, episode_count)
                self.summary_writer.flush()
                                
            episode_count += 1
            
            # entropy annealing
            entr_ = entr_ - 1/self.n_iterations ####Change IP
            
                                                  

class neural_network:
    '''
    neural_network, that includes methods to train and test a reinforcement learning model
    '''    
    def __init__(self 
                 , bandit  # The bandit object that defines the task for the agent to learn.
                 , noise = 'none'  # The type of noise to apply to the agent's actions during training.
                 , noise_parameter = 0  # The strength of the noise to apply.
                 , entropy_loss_weight = 0  # The weight of the entropy loss in the overall loss function.
                 , value_loss_weight = 0  # The weight of the value loss in the overall loss function.
                 , rnn_type = 'rnn'  # The type of recurrent neural network to use.
                 , learning_algorithm = 'reinforce'  # The algorithm to use for learning.
                 , discount_rate = 0.5  # The discount rate to use for computing the reward.
                 , learning_rate = 1e-4  # The learning rate for the optimization algorithm.
                 , n_hidden_neurons = 48  # The number of neurons to use in the hidden layer of the neural network.
                 , n_iterations = 50000  # The number of iterations to train the model for.
                 , path_to_save_model = 'saved_models/'  # The path to save the trained model to.
                 , path_to_save_progress = 'tensorboard/'  # The path to save the model's training progress to.
                 , path_to_save_test_files = 'data/rnn_raw_data/'  # The path to save the results of the model's testing to.
                 , model_id = 0  # An identifier for the model.
                 ):
        
        # Set the input parameters as attributes of the object.
        self.bandit = bandit
        self.noise = noise
        self.noise_parameter = noise_parameter
        self.entropy_loss_weight = entropy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.rnn_type = rnn_type
        self.learning_algorithm = learning_algorithm
        self.discount_rate      = discount_rate
        self.learning_rate      = learning_rate
        self.n_hidden_neurons = n_hidden_neurons
        self.n_iterations = n_iterations
        self.path_to_save_model = path_to_save_model
        self.path_to_save_progress = path_to_save_progress
        self.path_to_save_test_files = path_to_save_test_files
        self.model_id = model_id
                           
        # Apply the noise to the agent's actions during training, if specified.
        if self.noise == 'update-dependant':
            self.noise_parameter = self.noise_parameter
        if self.noise == 'constant':
            raise ValueError('constant noise is not implemented yet!')
        if self.noise == 'none':
            self.noise_parameter = 0
            
        # Create a name for the model based on its input parameters.
        
        self.model_name = '{}_{}_nh_{}_lr_{}_n_{}_p_{}_ew_{}_vw_{}_dr_{}_{}_d_{}_p_{}_rt_{}_a_{}_n_{}_te_{}_id_{}'.format(self.rnn_type
                                                                , self.learning_algorithm[0:3]
                                                                , self.n_hidden_neurons
                                                                , dot2_(self.learning_rate, is_lr = True)
                                                                , self.noise[0]
                                                                , dot2_(self.noise_parameter)
                                                                , dot2_(self.entropy_loss_weight)
                                                                , dot2_(self.value_loss_weight)
                                                                , dot2_(self.discount_rate)
                                                                , self.bandit.bandit_type[0:3]
                                                                , str(self.bandit.dependant)[0]
                                                                , dot2_(self.bandit.bandit_parameter) 
                                                                , self.bandit.reward_type[0:3]
                                                                , self.bandit.arms
                                                                , self.bandit.num_steps
                                                                , self.n_iterations
                                                                , self.model_id).lower()    
        
        self.model_path = self.path_to_save_model + self.model_name
        
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate) 

    def train(self):
            
        # train the RNN
        train = True
        
        # create folder to save progress
        if not os.path.exists(self.path_to_save_progress):
            os.makedirs(self.path_to_save_progress)
            
        # create the graph
        self.worker = Worker(game=conditioning_bandit(self.bandit),
                             trainer=self.trainer,
                             model_path=self.model_path,
                             model_name=self.model_name,
                             noise=self.noise,
                             path_to_save_progress=self.path_to_save_progress,
                             n_hidden_neurons=self.n_hidden_neurons,
                             n_arms=self.bandit.arms,
                             num_steps=self.bandit.num_steps,
                             n_iterations=self.n_iterations,
                             rnn_type=self.rnn_type,
                             noise_parameter=self.noise_parameter,
                             entropy_loss_weight=self.entropy_loss_weight,
                             value_loss_weight=self.value_loss_weight,
                             learning_algorithm=self.learning_algorithm)
        
        # create the saver
        self.saver = tf.train.Saver(max_to_keep=5)
        
        # start tf.Session
        with tf.Session() as sess:
            
            # if model exists start from the last checkpoint
            if os.path.exists(self.model_path):
                print('Resuming Training Model: {}'.format(self.model_name))
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                print(ckpt)
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                # train
                self.worker.work(self.discount_rate, sess, self.saver, train)
                    
            else: 
                print('Training Model: {}'.format(self.model_name))
                # initialise variables
                sess.run(tf.global_variables_initializer())
                # train
                self.worker.work(self.discount_rate, sess, self.saver, train)
                
        # reset the graph
        self.reset()
        
        
    def test(self, n_runs, bandit_param_range, bandit, num_rins=1, path_to_fixed_bandits='data/intermediate_data/fixed_bandits/'):
    
        # Set the train flag to False to avoid training the RNN.
        train = False
    
        # Set the value of the train standard deviation parameter to the bandit's standard deviation.
        train_sd = self.bandit.bandit_parameter
        
        # Loop over the bandit parameter range.
        for sd_ in bandit_param_range:
            
            # Create an empty list to store the test dataframes.
            df_list = []
            
            # Set the bandit's standard deviation parameter to the current value of sd_.
            temp_sd = dot2_(sd_)
            
            # Loop over the number of runs.
            for run in range(n_runs): 
                
                # Define the name of the current bandit zip file.
                bandit_zip_name = bandit.format(temp_sd, str(run))
                
                # Loop over the number of reward instances.
                for rin in range(num_rins):
                    
                    # If the bandit is a string and contains 'Daw2006', load the DAW bandit from a CSV file.
                    if isinstance(bandit, str) and 'Daw2006' in bandit:
                        
                        # Load the DAW bandit from a CSV file.
                        df = pd.read_csv(bandit)
                        
                        # Convert the dataframe into a bandit class.
                        self.bandit = fdbc.load_daw_bandit(df)
                        
                    else: # If the bandit is not a DAW bandit.
                
                        # Extract the current fixed test bandit from the zip file.
                        bandit_zip = zip2csv(path_to_data=path_to_fixed_bandits, zip_file_name=bandit_zip_name)
                        bandit_file_name = bandit_zip_name.replace('.zip', '_rin_{}.csv'.format(str(rin)))
                        bandit_zip.extract_file(bandit_file_name)
                        fixed_test_bandit = pd.read_csv(bandit_file_name)
                        
                        # Convert the dataframe into a bandit class.
                        self.bandit = fbc.load_bandit(fixed_test_bandit)
                        
                        # Assign the current value of sd_ to the bandit's standard deviation parameter.
                        self.bandit.bandit_parameter = sd_
        
                        # Delete the presaved bandit.
                        bandit_zip.delete_file(bandit_file_name)
                            
                    # Test the RNN.
                    # Create the graph.
                    self.worker  = Worker(game=conditioning_bandit(self.bandit),
                                          trainer=self.trainer, model_path=self.model_path,
                                          model_name=self.model_name, noise=self.noise,
                                          path_to_save_progress=self.path_to_save_progress,
                                          n_hidden_neurons=self.n_hidden_neurons,
                                          n_arms=self.bandit.arms, num_steps=self.bandit.num_steps,
                                          n_iterations=self.n_iterations, rnn_type=self.rnn_type,
                                          noise_parameter=self.noise_parameter,
                                          entropy_loss_weight=self.entropy_loss_weight,
                                          value_loss_weight=self.value_loss_weight,
                                          learning_algorithm=self.learning_algorithm)
                    
                    # Create a saver.
                    self.saver = tf.train.Saver(max_to_keep=5)

                    # Start a TensorFlow session
                    with tf.Session() as sess:
                        
                        # Get the checkpoint state of the model from the model path
                        ckpt = tf.train.get_checkpoint_state(self.model_path)
                        
                        # Print the model path
                        print(self.model_path)
                        
                        # Restore the saved model to the current session
                        self.saver.restore(sess,ckpt.model_checkpoint_path)
                        
                        # Get a test dataframe
                        df = self.worker.work(self.discount_rate, sess, self.saver, train)
                        
                        '''
                        Create a multi-indexed dataframe and save as pickle
                        '''
                        
                        # Add columns that will be used as indices
                        df['rnn_type'] = self.rnn_type
                        df['noise'] = self.noise
                        df['entropy_loss_weight'] = self.entropy_loss_weight
                        df['value_loss_weight'] = self.value_loss_weight
                        df['learning_algorithm'] = self.learning_algorithm
                        df['train_sd'] = train_sd
                        df['run'] = run
                        df['reward_instance'] = rin
                        df['test_sd'] = self.bandit.bandit_parameter
                        df['rnn_id'] = self.model_id
                        
                        # Calculate the accuracy of the model on the test data
                        accuracy = [int(ch == np.argmax([p1, p2, p3, p4])) for ch, p1, p2, p3, p4 in zip(df['choice'], df['p_rew_1'], df['p_rew_2'], df['p_rew_3'], df['p_rew_4'])]
                        df['accuracy'] = accuracy
                        
                        # Determine if the subject switched choices
                        is_switch = [int(df.choice[t] != df.choice[t-1]) for t in range(1, len(df['choice']))]
                        is_switch = np.append(0, is_switch)
                        df['is_switch'] = is_switch
                        
                        # Append the dataframe to the list of dataframes
                        df_list.append(df)
                        
                        # Print the mean accuracy of the model on the test data
                        print('ACCURACY')
                        print(np.mean(df['accuracy']))
                    
                    # reset graph
                    self.reset()
                    
                    # create list with names of the index of the multiindex df
                    multiindex_list = ['rnn_type', 'learning_algorithm', 'noise',  'train_sd'
                                       ,'rnn_id', 'test_sd', 'run', 'reward_instance']
                    
                    # concat df_list rowwise
                    all_dfs = pd.concat(df_list)
                    
                    # make all_dfs a muliindex df
                    mult_ind_df = all_dfs.set_index(multiindex_list)
                    
                    # pickle the file
                    filename = self.path_to_save_test_files + self.model_name + '_test_b_{}_p_{}'.format(self.bandit.bandit_type[0:3], temp_sd)
                                
                    outfile = open(filename,'wb')
                    pickle.dump(mult_ind_df, outfile)
                    outfile.close()
                    
                    print('FINISHED')
                    
    def reset(self):
        tf.reset_default_graph()
                    
                                             
