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



"""

Neural Network Class

methods: 
    
    train

        Input: 
            bandit 
            number of hidden neurons
            number of samples
            noise = bol
            weber_fraction
            entropy_loss_weight
            path_to_save_model
            path_to_save_progress
            
        Output:
            save model
            track progress on tensorboard
            
    test
    
        Input:
            bandit
            n_runs
            path_to_save
            
        Output: 
            save .csv files and compress
"""


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

# class to define the graph
class AC_Network():
    def __init__(self, trainer, noise, rnn_type, noise_parameter
                 , n_hidden_neurons, n_arms, entropy_loss_weight ####Change IP
                 , value_loss_weight, learning_algorithm):
        '''
        Returns the graph. 
        Takes as input: trainer, a tensorflow optimizer
                        noise, with computation noise (noise=1) or decision entropy (noise=0)
                        coefficient, coefficient for the computation noise or decision entropy

        '''
        
        # Input
        self.prev_rewardsch        = tf.placeholder(shape=[None,1], dtype=tf.float32, name="v1")
        self.prev_actions          = tf.placeholder(shape=[None], dtype=tf.int32 , name="v2")
        self.prev_actions_onehot   = tf.one_hot(self.prev_actions, n_arms, dtype=tf.float32 , name="v3") #changed
        self.timestep              = tf.placeholder(shape=[None,1], dtype=tf.float32, name="v4")
        input_                     = tf.concat([self.prev_rewardsch, self.prev_actions_onehot],1, name="v5")

        self.actions             = tf.placeholder(shape=[None], dtype=tf.int32, name = "v6")
        self.actions_onehot      = tf.one_hot(self.actions, n_arms, dtype=tf.float32,  name = "v7")
        
        if entropy_loss_weight == 'linear':
            self.entropy_loss_weight = tf.placeholder("float", None,  name = "v8") ####Change IP
            
        else: 
            self.entropy_loss_weight = entropy_loss_weight
            
        print('entropy loss weight')
        print(self.entropy_loss_weight)

        #Recurrent network for temporal dependencies
        nb_units = n_hidden_neurons
        
        if rnn_type == 'lstm' and noise == 'none': # add rnn_type argument
            
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(nb_units,state_is_tuple=True)
        
            # # added for fixed point analysis
            # self.lstm_cell = lstm_cell
            
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(input_, [0]) ##unsure about this
            step_size = tf.shape(self.prev_rewardsch)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,time_major=False)
            lstm_c, lstm_h = lstm_state
            
            # add added_noises_means for reinforce algorithm
            self.added_noises_means = tf.convert_to_tensor(np.zeros(48))
            
            #print('shapes of ht and ct')
            # print(np.shape(lstm_h))
            # print(np.shape(lstm_c))
            
            #print('sliced c')
            # print(np.shape(lstm_c[:1, :]))
            
            # added noise, does it work?
            self.h_noise    = tf.placeholder(tf.float32, [None, nb_units])   
            
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            #print('lstm state out')
            # print(np.shape(self.state_out))
            # print(np.shape(self.state_out[0]))
            # print(np.shape(self.state_out[1]))
            
            self.true_state_out = lstm_h[:1, :]
            rnn_out = tf.reshape(lstm_outputs, [-1, nb_units])
            
            #print('lstm rnn_out (lstm_outputs) before reshape')
            # print(np.shape(lstm_outputs))
            
            #print('lstm rnn_out')
            # print(np.shape(rnn_out))
            
        if rnn_type == 'lstm2' and noise == 'update-dependant':
            
            add_noise = True
            
            # IMPLEMENT SCAN HERE
            lstm_cell_b = tf.contrib.rnn.BasicLSTMCell(nb_units,state_is_tuple=True) #hacky

            lstm_cell = own_lstm_cell.LSTM(n_arms+1, nb_units, add_noise)
            c_init = np.zeros((1, nb_units), np.float32)
            h_init = np.zeros((1, nb_units), np.float32)
            self.state_init = [h_init, c_init]
            c_in = tf.placeholder(tf.float32, [1, nb_units])
            h_in = tf.placeholder(tf.float32, [1, nb_units])
            self.state_in = tf.tuple([h_in, c_in])
            
            self.h_noise    = tf.placeholder(tf.float32, [None, nb_units]) 
            all_noises      = self.h_noise
            
            # print('all_noises')
            # print(np.shape(all_noises))
            
            all_inputs = tf.concat((input_, all_noises), axis = 1) 
            rnn_in = tf.transpose(tf.expand_dims(all_inputs, [0]), [1,0,2])
            
            # print('rnn_in')
            # print(np.shape(rnn_in))
            
            
            # states, self.added_noises_means = tf.scan(lstm_cell.step, rnn_in, initializer=(self.state_in))
            states = tf.scan(lstm_cell.step, rnn_in, initializer=(self.state_in))
            
            # add added_noises_means for reinforce algorithm
            self.added_noises_means = tf.convert_to_tensor(np.zeros(48))
            
            lstm_h, lstm_c = states
            
            # print('lstm_h')
            # print(np.shape(lstm_h))
            
            # print('lstm_c')
            # print(np.shape(lstm_c))

            self.state_out = (lstm_h[0,:, :], lstm_c[0,:, :]) # Deniz: changed to get [1, n_hidden] vs [?, 1, n_hidden] which throws error in recursion in work
            
            # print('state_out')
            # print(np.shape(self.state_out))
            # print('state_out lstm_h')
            # print(np.shape(self.state_out[0]))
            # print('state_out lstm_c')
            # print(np.shape(self.state_out[1]))
            
            self.true_state_out = lstm_h[:1, :]
            # print('true state_out')
            # print(np.shape(self.true_state_out))
            
            rnn_out        = tf.reshape(lstm_h, [-1, nb_units])
            
            # print('rnn_out')
            # print(np.shape(rnn_out))
        
        if rnn_type == 'rnn': 
    
            if noise == 'update-dependant':
                add_noise = True
            if noise == 'none':
                add_noise = False
            if noise == 'constant':
                raise  ValueError('Constant Noise in RNN not implemented yet!')
            
            lstm_cell       = recurrent_networks.RNN(n_arms+1, nb_units, add_noise) # input (last reward, one-hot actions)
            h_init          = np.zeros((1, nb_units), np.float32)
            self.state_init = [h_init]        
            self.h_in       = tf.placeholder(tf.float32, [1, nb_units])        
            self.h_noise    = tf.placeholder(tf.float32, [None, nb_units])        
            self.state_in   = self.h_in
            all_noises      = self.h_noise
    
            if add_noise: 
                all_inputs         = tf.concat((input_, all_noises), axis=1)
                rnn_in             = tf.transpose(tf.expand_dims(all_inputs, [0]),[1,0,2])
            else:
                rnn_in = tf.transpose(tf.expand_dims(input_, [0]),[1,0,2])
                
            # print('input_ shape')
            # print(np.shape(input_))
                
            # print('rnn_in shape findling')
            # print(np.shape(rnn_in))
            
            states, self.added_noises_means    = tf.scan(lstm_cell.step, rnn_in, initializer=(self.state_in, 0.))
            
            # print('findling states')
            # print(np.shape(states))
            
            lstm_h         = states[:,0]
            # print('lstm_h shape')
            # print(np.shape(lstm_h))
            
            self.state_out = states[:1,0]
            # print('state_out shape findling')
            # print(np.shape(self.state_out))
            
            self.true_state_out = self.state_out
            rnn_out        = lstm_h
        
        # Loss functions
        
        self.policy = slim.fully_connected(rnn_out, n_arms, activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)   #changed   
        
        # print('softmax')
        # print(np.shape(self.policy))
        
        self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
       
        self.entropy     = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
        
        self.policy_loss = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.advantages)
        
        self.loss_entropy = self.entropy * self.entropy_loss_weight
       
        if learning_algorithm == 'a2c':
        
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None) # added AC
                        
            self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)# added AC
            
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1]))) # add value loss weight

            self.loss        = value_loss_weight *self.value_loss + self.policy_loss - self.loss_entropy # add entropy_loss_weight
            
        if learning_algorithm == 'reinforce':
            
            self.loss        = self.policy_loss - self.loss_entropy
        
        #Get gradients from network using losses
        local_vars            = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        self.gradients        = tf.gradients(self.loss,local_vars)
        # added 
        self.gradient_norm    = tf.global_norm(self.gradients)
        
        self.var_norms        = tf.global_norm(local_vars)         
        self.apply_grads      = trainer.apply_gradients(zip(self.gradients,local_vars))
        
class Worker():
    def __init__(self, game, trainer, model_path, model_name, noise
                 , path_to_save_progress, n_hidden_neurons, n_arms, num_steps
                 , n_iterations, rnn_type, noise_parameter, entropy_loss_weight    ####Change IP
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
        train method
        '''        
        rollout           = np.array(rollout)
        actions           = rollout[:,0]
        rewards_ch        = rollout[:,1]
        timesteps         = rollout[:,2]
        h_noises          = rollout[:,3]

        prev_actions      = [2] + actions[:-1].tolist()    # initialize one-hot vector representing the previous chosen action of episode to 0
        prev_rewards_ch   = [0] + rewards_ch[:-1].tolist() # initialize previous observed reward of episode to 0
        
        if self.learning_algorithm == 'a2c':
            values = rollout[:,4]
            
            self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
            advantages = rewards_ch + gamma * self.value_plus[1:] - self.value_plus[:-1]
            advantages = discount(advantages,gamma) # added AC
            
        
        
        if self.learning_algorithm == 'reinforce':
            values = tf.convert_to_tensor(np.zeros(300))
        
        self.rewards_plus  = np.asarray(rewards_ch.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]

       
        if self.rnn_type == 'lstm' and self.learning_algorithm == 'a2c' or self.rnn_type == 'lstm2' and self.learning_algorithm == 'a2c':
            
            rnn_state = self.ac_network.state_init ###change
            
            
            if self.entropy_loss_weight == 'linear':
                
                feed_dict = {self.ac_network.target_v:discounted_rewards,
                             self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                             self.ac_network.prev_actions:prev_actions,
                             self.ac_network.h_noise:np.vstack(h_noises), 
                             self.ac_network.actions:actions,
                             self.ac_network.timestep:np.vstack(timesteps),
                             self.ac_network.advantages:advantages,
                             self.ac_network.state_in[0]:rnn_state[0],
                             self.ac_network.state_in[1]:rnn_state[1],
                             self.ac_network.entropy_loss_weight: entr_}   
                
            else:
                
                feed_dict = {self.ac_network.target_v:discounted_rewards,
                              self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                              self.ac_network.prev_actions:prev_actions,
                              self.ac_network.h_noise:np.vstack(h_noises),
                              self.ac_network.actions:actions,
                              self.ac_network.timestep:np.vstack(timesteps),
                              self.ac_network.advantages:advantages,
                              self.ac_network.state_in[0]:rnn_state[0],
                              self.ac_network.state_in[1]:rnn_state[1]} 
                
            v_l, p_l,e_l,v_n,_, grad_ = sess.run([self.ac_network.value_loss,
                                      self.ac_network.policy_loss,
                                      self.ac_network.entropy,
                                      self.ac_network.var_norms,
                                      self.ac_network.apply_grads, 
                                      self.ac_network.gradient_norm],
                                      feed_dict=feed_dict)            

            return v_l / len(rollout), p_l / len(rollout),e_l / len(rollout), 0.,v_n, grad_ # added AC

                
        if self.rnn_type == 'lstm2' and self.learning_algorithm == 'reinforce':
            
            rnn_state = self.ac_network.state_init ###change
            
            
            if self.entropy_loss_weight == 'linear':
                
                
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
                feed_dict = {self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                             self.ac_network.prev_actions:prev_actions,
                             self.ac_network.h_noise:np.vstack(h_noises),
                             self.ac_network.actions:actions,
                             self.ac_network.timestep:np.vstack(timesteps),
                             self.ac_network.advantages:discounted_rewards,
                             self.ac_network.state_in[0]:rnn_state[0],
                             self.ac_network.state_in[1]:rnn_state[1]} 
                

            p_l,e_l,v_n,_, grad_ = sess.run([self.ac_network.policy_loss,
                                                  self.ac_network.entropy,
                                                  self.ac_network.var_norms,
                                                  self.ac_network.apply_grads, 
                                                  self.ac_network.gradient_norm],
                                                  feed_dict=feed_dict)

        
            return p_l / len(rollout),e_l / len(rollout), 0.,v_n, grad_ # added AC

        
        if self.rnn_type == 'lstm' and self.learning_algorithm == 'reinforce':
                        
            rnn_state = self.ac_network.state_init ###change
                        
            if self.entropy_loss_weight == 'linear':
                
                
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
                feed_dict = {self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch),
                             self.ac_network.prev_actions:prev_actions,
                             self.ac_network.h_noise:np.vstack(h_noises),
                             self.ac_network.actions:actions,
                             self.ac_network.timestep:np.vstack(timesteps),
                             self.ac_network.advantages:discounted_rewards,
                             self.ac_network.state_in[0]:rnn_state[0],
                             self.ac_network.state_in[1]:rnn_state[1]} 
            
            

            p_l,e_l,v_n,_, grad_ = sess.run([self.ac_network.policy_loss,
                                             self.ac_network.entropy,
                                             self.ac_network.var_norms,
                                             self.ac_network.apply_grads, 
                                             self.ac_network.gradient_norm],
                                             feed_dict=feed_dict)
            
            return p_l / len(rollout),e_l / len(rollout), 0.,v_n, grad_
        
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
        if train == False:
            
            # prepare colnames for p_rew, softmax, unit activity and unit noise
            rnn_state_col_names = ['']*self.n_hidden_neurons
            rnn_state_noise_col_names = ['']*self.n_hidden_neurons
            
            rnn_prob_rew = ['']*self.n_arms
            rnn_softmax = ['']*self.n_arms
            
            for i in range(self.n_hidden_neurons):
                rnn_state_col_names[i] = 'rnn_state_'+str(i+1)
                rnn_state_noise_col_names[i] = 'added_noise_rnn_state_'+str(i+1)
                
            for i in range(self.n_arms):
                rnn_prob_rew[i] = 'p_rew_'+str(i+1)
                rnn_softmax[i] = 'softmax_'+str(i+1)           
            
            # prepare colname for bandit parameter
            if self.env.game.bandit_type == 'restless':
                # bandit_par = self.env.game.bandit_parameter  # have to change this
                bandit_par = 'sd_noise'
            if self.env.game.bandit_type == 'stationary':
                # bandit_par = self.env.game.bandit_parameter  # have to change this
                bandit_par = 'p_rew_best'
            else: 
                bandit_par = 'bandit parameter'
                
            colnames = [bandit_par,'choice', 'reward', 'value']
            
            colnames.extend(rnn_prob_rew + rnn_softmax + rnn_state_col_names + rnn_state_noise_col_names)
            
            df = pd.DataFrame(columns=colnames)
            
            # prepare variables to collect data
            my_a = []
            my_rch = []
            my_r_prob = np.zeros([self.num_steps, self.n_arms])
            rnn_softmax_arr = np.zeros([self.num_steps, self.n_arms])
            rnn_state_arr = np.zeros([ self.num_steps, self.n_hidden_neurons])
            rnn_value_arr = np.zeros([self.num_steps]) # should be working?
            rnn_state_noise_arr = np.zeros([self.num_steps, self.n_hidden_neurons])
        ##################################################################################
        
        # take episode count if model was already trained
        if os.path.exists(self.model_path):
            list_of_files = glob.glob(self.model_path + '/*')
            print('###############')
            list_of_files = np.array(list_of_files)[[not "checkpoint" in s for s in list_of_files]] # list ignoring checkpoint file
            print(list_of_files)
            latest_file = max(list_of_files, key=os.path.getctime)
            print(latest_file)
            x = latest_file
            episode_count = int(x.split('-')[1].split('.')[0]) # get episode_count from latest file
            print(episode_count)
        else:
            episode_count = 0

        entr_ = 1


        while True:
            episode_buffer, state_mean_arr, added_noise_arr = [], [], []
            episode_reward, episode_step_count = 0, 0
            d, a, t, rch       = False, 2, 0, 0 #initialization parameters (in particular, the previous action is initialized to a null one-hot vector, a=2)
            
            
            if self.rnn_type == 'lstm':
                rnn_state          = self.ac_network.state_init
                
            if self.rnn_type == 'lstm2':
                rnn_state          = self.ac_network.state_init                
            
            if self.rnn_type == 'rnn':
                rnn_state          = self.ac_network.state_init[0]
                   
            self.env.reset()
            
            while d == False:
                
                if self.noise == 'update-dependant':
                    
                    if self.rnn_type == 'lstm2':
                        h_noise = np.array(np.random.normal(size=rnn_state[0].shape) * self.noise_parameter, dtype=np.float32)                    
                    
                    else: 
                        h_noise = np.array(np.random.normal(size=rnn_state.shape) * self.noise_parameter, dtype=np.float32)                    
                    
                if self.noise == 'constant':
                    raise  ValueError('Constant noise not implemented yet!')
                
                if self.noise == 'none':
                    h_noise = np.array(np.random.normal(size=self.ac_network.state_init[0].shape) * self.noise_parameter, dtype=np.float32)
                
                
                #Take an action using probabilities from policy network output.
                
                if self.rnn_type == 'lstm':
                    
                    if self.entropy_loss_weight == 'linear':
                        
                        feed_dict = {self.ac_network.prev_rewardsch:[[rch]],
                                     self.ac_network.prev_actions:[a],
                                     self.ac_network.timestep:[[t]],
                                     self.ac_network.state_in[0]:rnn_state[0],
                                     self.ac_network.state_in[1]:rnn_state[1],
                                     self.ac_network.h_noise:h_noise, 
                                     self.ac_network.entropy_loss_weight:entr_}
                        
                    else:
                        
                        feed_dict = {self.ac_network.prev_rewardsch:[[rch]],
                                     self.ac_network.prev_actions:[a],
                                     self.ac_network.timestep:[[t]],
                                     self.ac_network.state_in[0]:rnn_state[0],
                                     self.ac_network.state_in[1]:rnn_state[1],
                                     self.ac_network.h_noise:h_noise}
                        
                if self.rnn_type == 'lstm2':
                    
                    if self.entropy_loss_weight == 'linear':
                        
                        feed_dict = {self.ac_network.prev_rewardsch:[[rch]],
                                     self.ac_network.prev_actions:[a],
                                     self.ac_network.timestep:[[t]],
                                     self.ac_network.state_in[0]:rnn_state[0],
                                     self.ac_network.state_in[1]:rnn_state[1],
                                     self.ac_network.h_noise:h_noise, 
                                     self.ac_network.entropy_loss_weight:entr_}
                        
                    else:
                        
                        feed_dict = {self.ac_network.prev_rewardsch:[[rch]],
                                     self.ac_network.prev_actions:[a],
                                     self.ac_network.timestep:[[t]],
                                     self.ac_network.state_in[0]:rnn_state[0],
                                     self.ac_network.state_in[1]:rnn_state[1],
                                     self.ac_network.h_noise:h_noise}
                        
                
                if self.rnn_type == 'rnn':
                    
                    if self.entropy_loss_weight == 'linear':
                        
                        feed_dict = {self.ac_network.prev_rewardsch:[[rch]],
                                     self.ac_network.prev_actions:[a],
                                     self.ac_network.timestep:[[t]],
                                     self.ac_network.h_in:rnn_state,
                                     self.ac_network.h_noise:h_noise, 
                                     self.ac_network.entropy_loss_weight:entr_}
                        
                    else:
                        
                        feed_dict = {self.ac_network.prev_rewardsch:[[rch]],
                                     self.ac_network.prev_actions:[a],
                                     self.ac_network.timestep:[[t]],
                                     self.ac_network.h_in:rnn_state,
                                     self.ac_network.h_noise:h_noise}
                                        
                                    
                if self.learning_algorithm == 'a2c':
                
                    a_dist,v,rnn_state_new, rnn_true_state_new = sess.run([self.ac_network.policy,
                                                                           self.ac_network.value,
                                                                           self.ac_network.state_out,
                                                                           self.ac_network.true_state_out],
                                                                           feed_dict=feed_dict)
                    
                if self.learning_algorithm == 'reinforce':
                                        
                    a_dist,rnn_state_new, rnn_true_state_new, added_noise = sess.run([self.ac_network.policy,
                                                                             self.ac_network.state_out,
                                                                             self.ac_network.true_state_out,
                                                                             self.ac_network.added_noises_means], # removed added_noises_means, states_means 
                                                                             feed_dict=feed_dict)

                    # value = 0
                    v = np.array([[0]])
                
                
                a                   = np.random.choice(a_dist[0],p=a_dist[0])
                a                   = np.argmax(a_dist == a)
                
                if self.rnn_type == 'lstm':

                    rnn_state           = rnn_state_new
                    
                if self.rnn_type == 'lstm2':
                    
                    rnn_state           = rnn_state_new
                
                if self.rnn_type == 'rnn':
                    rnn_state           = rnn_state_new[:2]
                    
                rch,d,t             = self.env.pullArm(a)
                # for tensorboard: if rewards 1, 0
                # episode_reward     += rch
                # for tensorboard: if rewards 1, -1
                episode_reward     += rch
                
                episode_step_count += 1 

                # state_mean_arr.append(state_mean)
                #added_noise_arr.append(added_noise)
                episode_buffer.append([a,rch,t, h_noise, v[0,0], d])
                
                # if network is tested collect vaiables
                if train == False: 
                    my_a.append(a)
                    my_rch.append(rch)
                    my_r_prob[t] = self.env.reward_probs[t]
                    rnn_softmax_arr[t] = a_dist
                    rnn_value_arr[t] = v[0,0]
                    rnn_state_arr[t] = rnn_true_state_new[:2] # should be okay as rnn_true_state is rnn_state_new with rnn_type == 'rnn'
                    rnn_state_noise_arr[t] = h_noise
                
                if not d:
                    self.env.update()        
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_step_count)

            # Update the network using the experience buffer at the end of the episode.
            # added gg
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
    
    def __init__(self 
                 , bandit
                 , noise = 'none'
                 , noise_parameter = 0
                 , entropy_loss_weight = 0
                 , value_loss_weight = 0
                 , rnn_type = 'rnn'
                 , learning_algorithm = 'reinforce'
                 , discount_rate = 0.5
                 , learning_rate = 1e-4
                 , n_hidden_neurons = 48
                 , n_iterations = 50000
                 , path_to_save_model = 'saved_models/'
                 , path_to_save_progress = 'tensorboard/'
                 , path_to_save_test_files = 'data/rnn_raw_data/'
                 , model_id = 0
                 ):
        
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
        
        
        if self.noise == 'update-dependant':
            self.noise_parameter = self.noise_parameter
        
        if self.noise == 'constant':
            raise ValueError('constant noise is not implemented yet!')
        
        if self.noise == 'none':
            self.noise_parameter = 0

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
        
        self.model_name = self.model_name
        
        self.model_path = self.path_to_save_model + self.model_name
        
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate) 
        
    def train(self):
        
        # train the RNN
        train = True
                
        # create folder to save progress
        if not os.path.exists(self.path_to_save_progress):
            os.makedirs(self.path_to_save_progress)
        
        # create the graph
                
        self.worker  = Worker(game = conditioning_bandit(self.bandit)
                      , trainer = self.trainer, model_path = self.model_path
                      , model_name = self.model_name
                      , noise = self.noise
                      , path_to_save_progress = self.path_to_save_progress
                      , n_hidden_neurons = self.n_hidden_neurons
                      , n_arms = self.bandit.arms
                      , num_steps = self.bandit.num_steps
                      , n_iterations = self.n_iterations
                      , rnn_type = self.rnn_type
                      , noise_parameter = self.noise_parameter
                      , entropy_loss_weight = self.entropy_loss_weight
                      , value_loss_weight = self.value_loss_weight
                      , learning_algorithm = self.learning_algorithm)
        
        # create the saver
        self.saver   = tf.train.Saver(max_to_keep=5)
        
        # start tf.Session
        with tf.Session() as sess:
            
            # if model exists start from the last checkpoint
            if os.path.exists(self.model_path):
                
                print('Resuming Training Model: {}'.format(self.model_name))
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                print(ckpt)
                self.saver.restore(sess,ckpt.model_checkpoint_path)
                # train
                self.worker.work(self.discount_rate,sess,self.saver,train)
                
            else: 
                print('Training Model: {}'.format(self.model_name))
                # initialise variables
                sess.run(tf.global_variables_initializer())
                # train
                self.worker.work(self.discount_rate,sess,self.saver,train)
            
        # reset the graph
        self.reset()
    
        
    def test(self, n_runs, bandit_param_range, bandit, num_rins = 1
             , path_to_fixed_bandits = 'data/intermediate_data/fixed_bandits/'):
        
        # if bandit_param_range == None:
        #     bandit_param_range = [bandit.bandit_parameter]
        
        # do not train the RNN
        train = False
        
        # get train parameter
        train_sd = self.bandit.bandit_parameter
        
        # # code to use if fixed bandits are used
        # if use_fixed_bandits:
            
        for sd_ in bandit_param_range:
            
            # create df list
            df_list = []
            
            temp_sd = dot2_(sd_)
            
            for run in range(n_runs): 
                
                # declare zip name
                bandit_zip_name = bandit.format(temp_sd, str(run))
                # zip_name = self.model_name + '_' + bandit_zip_name
                                            
                for rin in range(num_rins):
                    
                    # get daw bandits if test_mab is str
                    if isinstance(bandit, str) and 'Daw2006' in bandit:
                        
                        df = pd.read_csv(bandit)
                        # convert datafame into bandit class
                        self.bandit = fdbc.load_daw_bandit(df)
                        
                    else: # if not daw bandits
                
                        # extract bandit
                        
                        bandit_zip = zip2csv(path_to_data = path_to_fixed_bandits, zip_file_name = bandit_zip_name)
                        
                        bandit_file_name = bandit_zip_name.replace('.zip', '_rin_{}.csv'.format(str(rin)))
                        
                        bandit_zip.extract_file(bandit_file_name)
                        
                        fixed_test_bandit = pd.read_csv(bandit_file_name)
                        
                        # convert datafame into bandit class
                        self.bandit = fbc.load_bandit(fixed_test_bandit)
                        
                        # assign sd
                        self.bandit.bandit_parameter = sd_
    
                        # delete presaved bandit
                        bandit_zip.delete_file(bandit_file_name)
                        
                    # test rnn
                    # create the graph
                    self.worker  = Worker(game = conditioning_bandit(self.bandit)
                                  , trainer = self.trainer, model_path = self.model_path
                                  , model_name = self.model_name
                                  , noise = self.noise
                                  , path_to_save_progress = self.path_to_save_progress
                                  , n_hidden_neurons = self.n_hidden_neurons
                                  , n_arms = self.bandit.arms
                                  , num_steps = self.bandit.num_steps
                                  , n_iterations = self.n_iterations
                                  , rnn_type = self.rnn_type
                                  , noise_parameter = self.noise_parameter
                                  , entropy_loss_weight = self.entropy_loss_weight
                                  , value_loss_weight = self.value_loss_weight
                                  , learning_algorithm = self.learning_algorithm)
            
                    # create saver
                    self.saver   = tf.train.Saver(max_to_keep=5)
                
                    with tf.Session() as sess:
                        # print('Testing Model: {}'.format(self.model_name))
                        ckpt = tf.train.get_checkpoint_state(self.model_path)
                        
                        print(self.model_path)
                        
                        self.saver.restore(sess,ckpt.model_checkpoint_path)
                        
                        # get test dataframe                       
                        df = self.worker.work(self.discount_rate,sess,self.saver,train)

                        # file_name = zip_name.replace('.zip', '') +'_rin_{}.csv'.format(str(rin)).lower()
                        
                        '''
                        create multiindex df and save as pickle
                        '''
                        
                        # add columns later used as index
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
                        
                        accuracy = [int(ch==np.argmax([p1,p2,p3,p4])) for ch, p1, p2, p3, p4 in zip(df['choice'], df['p_rew_1'], df['p_rew_2'], df['p_rew_3'], df['p_rew_4'])]
                        df['accuracy'] = accuracy
                        
                        is_switch = [int(df.choice[t] != df.choice[t-1]) for t in range(1, len(df['choice']))]
                        is_switch = np.append(0, is_switch)
                        df['is_switch'] = is_switch
                        
                        # add df to df_list
                        
                        df_list.append(df)
                        
                        print('ACCURACY')
                        print(np.mean(df['accuracy']))
                        
                        
                    
                    # reset graph
                    self.reset()
        
            # create list with names of the index of the multiindex df
            multiindex_list = ['rnn_type', 'learning_algorithm', 'noise',  'train_sd' ####Change IP
                               ,'rnn_id', 'test_sd', 'run', 'reward_instance']
            
            # concat df_list rowwise
            all_dfs = pd.concat(df_list)
            
            # make all_dfs a muliindex df
            mult_ind_df = all_dfs.set_index(multiindex_list)
            
            # pickle the file
            # filename = self.path_to_save_test_files + 'all_{}_test_runs_train_sd_{}_id_{}_test_sd_{}'.format(reward_type, self.train_sd, self.model_id, temp_sd)
            
            filename = self.path_to_save_test_files + self.model_name + '_test_b_{}_p_{}'.format(self.bandit.bandit_type[0:3], temp_sd)
                        
            outfile = open(filename,'wb')
            pickle.dump(mult_ind_df, outfile)
            outfile.close()
            
            print('FINISHED')

    def reset(self):
        tf.reset_default_graph()
        



