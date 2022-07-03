import tensorflow as tf
import numpy as np

class LSTM():
    
    def __init__(self, input_size, hidden_size, add_noise):
        xav_init         = tf.contrib.layers.xavier_initializer
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.W           = tf.get_variable('W', shape=[4, hidden_size, hidden_size], initializer=xav_init())
        self.U           = tf.get_variable('U', shape=[4, input_size, hidden_size], initializer=xav_init())
        self.b_i           = tf.get_variable('b_i', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        self.b_f           = tf.get_variable('b_f', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        self.b_o           = tf.get_variable('b_o', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        self.b_g           = tf.get_variable('b_g', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        self.add_noise   = add_noise
        
    def step(self, prev, x):
        
        print('prev')
        print(np.shape(prev))
        
        
        
        ht_1, ct_1 = prev
        
        print('ht_1')
        print(np.shape(ht_1))
        
        print('ct_1')
        print(np.shape(ct_1))
        
        if self.add_noise:
            print('x')
            print(np.shape(x))
            
            
            x, noise_h = x[:,:self.input_size], x[:,-self.hidden_size:]
            
            print('x after')
            print(np.shape(x))
            
            print('noise_h')
            print(np.shape(noise_h))
        
        
        ####
        # GATES
        #
        #  input gate
        i = tf.sigmoid(tf.matmul(x,self.U[0]) + tf.matmul(ht_1,self.W[0]) + self.b_i)
        #  forget gate
        f = tf.sigmoid(tf.matmul(x,self.U[1]) + tf.matmul(ht_1,self.W[1]) + self.b_f)
        #  output gate
        o = tf.sigmoid(tf.matmul(x,self.U[2]) + tf.matmul(ht_1,self.W[2]) + self.b_o)
        #  gate weights
        g = tf.tanh(tf.matmul(x,self.U[3]) + tf.matmul(ht_1,self.W[3]) + self.b_g)
        ###
        # new internal cell state
        ct = ct_1*f + g*i
        # output state
        ht = tf.tanh(ct)*o
        ht_exact = ht
        
        # add noise here to ht
        if self.add_noise:
            
            noise_added = noise_h * tf.math.abs(ht - ht_1)
            print('noise_added')
            print(np.shape(noise_added))
            
            noise_added = tf.stop_gradient(noise_added) # treat noise as contraint, which the network is unaware of
            print('noise_added2')
            print(np.shape(noise_added))
            
            ht          = ht + noise_added
            
            print('ht_with_noise')
            print(np.shape(ht))
            
            print('tuple')
            print(np.shape(tf.tuple([ht, ct])))
            
            print('mean')
            print(tf.reduce_mean(tf.math.abs(ht - ht_exact)))
        
        return tf.tuple([ht, ct])#, tf.reduce_mean(tf.math.abs(ht - ht_exact))