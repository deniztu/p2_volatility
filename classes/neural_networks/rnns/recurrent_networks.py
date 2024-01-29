import tensorflow as tf

# # Elman dynamics

class RNN:
    """
    Class for custom Recurrent Neural Network (RNN) cell

    Parameters
    ----------
    input_size : int
        Size of the input.
    hidden_size : int
        Size of the hidden state.
    add_noise : bool
        Flag indicating whether to add noise during computation.

    Attributes
    ----------
    input_size : int
        Size of the input.
    hidden_size : int
        Size of the hidden state.
    W : Variable
        Weight matrix for the hidden state update.
    U : Variable
        Weight matrix for the input to hidden state transformation.
    b : Variable
        Bias term for the hidden state update.
    add_noise : bool
        Flag indicating whether to add noise during computation.

    Methods
    -------
    step(tuple_, x)
        RNN step function.

    """
    def __init__(self, input_size, hidden_size, add_noise):
        """
        Initialize the RNN cell instance.

        Parameters
        ----------
        input_size : int
            Size of the input.
        hidden_size : int
            Size of the hidden state.
        add_noise : bool
            Flag indicating whether to add noise during computation.
        """
        # Xavier weight initialization
        xav_init = tf.contrib.layers.xavier_initializer
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Weight matrix for the hidden state update
        self.W = tf.get_variable('W', shape=[hidden_size, hidden_size], initializer=xav_init())
        # Weight matrix for the input to hidden state transformation
        self.U = tf.get_variable('U', shape=[input_size, hidden_size], initializer=xav_init())
        # Bias term for the hidden state update
        self.b = tf.get_variable('b', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        # Flag indicating whether to add noise during computation
        self.add_noise = add_noise

    def step(self, tuple_, x):
        """
        RNN step function (forward pass).

        Parameters
        ----------
        tuple_ : tuple
            Tuple containing the previous hidden state and noise (if add_noise is True).
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Updated hidden state.
        Tensor
            Mean absolute difference between the updated hidden state and the exact hidden state.

        """
        st_1, _ = tuple_
        if self.add_noise:
            x, noise_h = x[:, :self.input_size], x[:, -self.hidden_size:]
        # Update hidden state
        ht = tf.matmul(st_1, self.W) + tf.matmul(x, self.U) + self.b
        ht_exact = ht
        if self.add_noise:
            noise_added = noise_h * tf.math.abs(ht - st_1)
            noise_added = tf.stop_gradient(noise_added)  # Treat noise as a constraint, which the network is unaware of
            ht = ht + noise_added
        return tf.tanh(ht), tf.reduce_mean(tf.math.abs(ht - ht_exact))

