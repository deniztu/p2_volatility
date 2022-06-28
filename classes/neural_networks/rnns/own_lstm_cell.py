import tensorflow as tf
import numpy as np

class LSTM():
    
    def __init__(self, input_size, hidden_size):
        xav_init         = tf.contrib.layers.xavier_initializer
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.W           = tf.get_variable('W', shape=[4, hidden_size, hidden_size], initializer=xav_init())
        self.U           = tf.get_variable('U', shape=[4, input_size, hidden_size], initializer=xav_init())
        # self.b_i           = tf.get_variable('b_i', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        # self.b_c           = tf.get_variable('b_c', shape=[hidden_size], initializer=tf.constant_initializer(0.))

        # self.add_noise   = add_noise
        
    def step(self, prev, x):
        
        # gather previous internal state and output state
        print('prev')
        print(np.shape(prev))
        
        print('x')
        print(np.shape(x))
        
        ht_1, ct_1 = prev
        ####
        # GATES
        #
        #  input gate
        i = tf.sigmoid(tf.matmul(x,self.U[0]) + tf.matmul(ht_1,self.W[0]))
        #  forget gate
        f = tf.sigmoid(tf.matmul(x,self.U[1]) + tf.matmul(ht_1,self.W[1]))
        #  output gate
        o = tf.sigmoid(tf.matmul(x,self.U[2]) + tf.matmul(ht_1,self.W[2]))
        #  gate weights
        g = tf.tanh(tf.matmul(x,self.U[3]) + tf.matmul(ht_1,self.W[3]))
        ###
        # new internal cell state
        ct = ct_1*f + g*i
        # output state
        ht = tf.tanh(ct)*o
        
        print('ht in step')
        print(np.shape(ht))
        
        # TODO add noise here to ht
        
        return tf.tuple([ht, ct])