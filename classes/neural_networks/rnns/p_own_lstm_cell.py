import tensorflow as tf

class LSTM:
    """
    Class for custom Long Short-Term Memory (LSTM) cell.

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
        Weight matrices for LSTM (for input, hidden state, and gates).
    U : Variable
        Weight matrices for LSTM (for input, hidden state, and gates).
    b_i : Variable
        Bias term for the input gate.
    b_f : Variable
        Bias term for the forget gate.
    b_o : Variable
        Bias term for the output gate.
    b_g : Variable
        Bias term for the gate weights.
    add_noise : bool
        Flag indicating whether to add noise during computation.

    Methods
    -------
    step(prev, x)
        LSTM step function.

    """
    def __init__(self, input_size, hidden_size, add_noise):
        """
        Initialize the LSTM instance.

        Parameters
        ----------
        input_size : int
            Size of the input.
        hidden_size : int
            Size of the hidden state.
        add_noise : bool
            Flag indicating whether to add noise during computation.
        """
        # Xavier initialization for weight matrices
        xav_init = tf.contrib.layers.xavier_initializer 

        # Assign input and hidden size to object
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices and biases for LSTM
        # U is for input, W is for hidden state
        # 4 because 4 different gates
        self.W = tf.get_variable('W', shape=[4, hidden_size, hidden_size], initializer=xav_init())
        self.U = tf.get_variable('U', shape=[4, input_size, hidden_size], initializer=xav_init())
        self.b_i = tf.get_variable('b_i', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        self.b_f = tf.get_variable('b_f', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        self.b_o = tf.get_variable('b_o', shape=[hidden_size], initializer=tf.constant_initializer(0.))
        self.b_g = tf.get_variable('b_g', shape=[hidden_size], initializer=tf.constant_initializer(0.))

        # Add noise or not
        self.add_noise = add_noise

    def step(self, prev, x):
        """
        LSTM step function (forward pass).

        Parameters
        ----------
        prev : tuple
            Tuple containing the previous hidden and cell states.
        x : Tensor
            Input tensor.

        Returns
        -------
        Tuple
            Tuple containing the updated hidden and cell states.

        """
        ht_1, ct_1 = prev  # Previous hidden and cell states

        if self.add_noise:
            x, noise_h = x[:, :self.input_size], x[:, -self.hidden_size:]  # Split input and noise

        # GATES
        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.U[0]) + tf.matmul(ht_1, self.W[0]) + self.b_i)
        # Forget gate
        f = tf.sigmoid(tf.matmul(x, self.U[1]) + tf.matmul(ht_1, self.W[1]) + self.b_f)
        # Output gate
        o = tf.sigmoid(tf.matmul(x, self.U[2]) + tf.matmul(ht_1, self.W[2]) + self.b_o)
        # Gate weights
        g = tf.tanh(tf.matmul(x, self.U[3]) + tf.matmul(ht_1, self.W[3]) + self.b_g)

        # New internal cell state
        ct = ct_1 * f + g * i
        # Output state
        ht = tf.tanh(ct) * o

        # Add noise here to ht
        if self.add_noise:
            noise_added = noise_h * tf.math.abs(ht - ht_1)  # Calculate noise to add
            noise_added = tf.stop_gradient(noise_added)  # Treat noise as constraint, which the network is unaware of
            ht = ht + noise_added  # Add noise to ht

        return tf.tuple([ht, ct])
