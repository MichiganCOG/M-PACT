""" FILE TO SUPPORT LAYER DEFINITIONS IN TENSORFLOW. ANY AND ALL LAYER DEFINITIONS USED TO MAKE NETWORKS MUST BE DERIVED FROM THIS DOCUMENT """

from tensorflow.contrib.rnn import static_rnn

import tensorflow as tf
import numpy      as np

def conv_layer(input_tensor,
               filter_dims,
               name,
               stride_dims = [1,1],
               weight_decay=0.0,
               padding='SAME',
               groups=1,
               trainable=True,
               non_linear_fn=tf.nn.relu,
               kernel_init=tf.truncated_normal_initializer(stddev=0.01),
               bias_init=tf.constant_initializer(0.1)):

    """
    Args:
        :input_tensor:  Input tensor to the convolutional layer
        :filter_dims:   A list detailing the height, width and number of channels for filters in the layer
        :stride_dims:   A list detailing the height and width of the stride between filters
        :name:          Scope name to be provided for current convolutional layer
        :padding:       Padding type definition (VALID or SAME)
        :non_linear_fn: Activation function applied to the outcome of the layer
        :kernel_init:   Tensorflow initialization function used to initialize the kernel
        :bias_init:     Tensorflow initialization function used to initialize the bias

    Return:
        :conv_out:      Output of the convolutional layer
    """

    input_dims = input_tensor.get_shape().as_list()

    # Ensure parameters match required shapes
    assert (len(input_dims) == 4)
    assert(len(filter_dims) == 3)
    assert(len(stride_dims) == 2)

    num_channels_in                      = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w                   = stride_dims

    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride_h, stride_w, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        if groups == 1:
            w = tf.get_variable('kernel', shape=[filter_h, filter_w, num_channels_in, num_channels_out],
                                initializer=kernel_init, regularizer=tf.contrib.layers.l2_regularizer(weight_decay), trainable=trainable)
            output = convolve(input_tensor, w)

        else:
            w = tf.get_variable('kernel', shape=[filter_h, filter_w, int(num_channels_in/groups), num_channels_out],
                                initializer=kernel_init, regularizer=tf.contrib.layers.l2_regularizer(weight_decay), trainable=trainable)

            input_groups  = tf.split(input_tensor, groups, axis=3)
            kernel_groups = tf.split(w, groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            output        = tf.concat(output_groups, 3)

        # END IF

        b        = tf.get_variable('bias', shape=[num_channels_out], initializer=bias_init, trainable=trainable)
        conv_out = output + b

        if non_linear_fn is not None:
            conv_out = non_linear_fn(conv_out, name=scope.name)

        # END IF

    # END WITH

    return conv_out


def conv3d_layer(input_tensor,
               filter_dims,
               name,
               stride_dims = [1,1,1],
               weight_decay=0.0,
               padding='SAME',
               groups=1,
               use_bias=True,
               trainable=True,
               non_linear_fn=tf.nn.relu,
               kernel_init=tf.truncated_normal_initializer(stddev=0.01),
               bias_init=tf.constant_initializer(0.1)):

    """
    Args:
        :input_tensor:  Input tensor to the convolutional layer
        :filter_dims:   A list detailing the depth, height, width and number of channels for filters in the layer
        :stride_dims:   A list detailing the depth, height and width of the stride between filters
        :name:          Scope name to be provided for current convolutional layer
        :padding:       Padding type definition (VALID or SAME)
        :non_linear_fn: Activation function applied to the outcome of the layer
        :kernel_init:   Tensorflow initialization function used to initialize the kernel
        :bias_init:     Tensorflow initialization function used to initialize the bias

    Return:
        :conv_out:      Output of the convolutional layer
    """

    input_dims = input_tensor.get_shape().as_list()

    # Ensure parameters match required shapes
    assert (len(input_dims) == 5)
    assert(len(filter_dims) == 4)
    assert(len(stride_dims) == 3)

    num_channels_in                                = input_dims[-1]
    filter_d, filter_h, filter_w, num_channels_out = filter_dims
    stride_d, stride_h, stride_w                   = stride_dims

    convolve = lambda i, k: tf.nn.conv3d(i, k, [1, stride_d, stride_h, stride_w, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        if groups == 1:
            w = tf.get_variable('kernel', shape=[filter_d, filter_h, filter_w, num_channels_in, num_channels_out],
                                initializer=kernel_init, regularizer=tf.contrib.layers.l2_regularizer(weight_decay), trainable=trainable)
            output = convolve(input_tensor, w)

        else:
            w = tf.get_variable('kernel', shape=[filter_d, filter_h, filter_w, int(num_channels_in/groups), num_channels_out],
                                initializer=kernel_init, regularizer=tf.contrib.layers.l2_regularizer(weight_decay), trainable=trainable)

            input_groups  = tf.split(input_tensor, groups, axis=3)
            kernel_groups = tf.split(w, groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            output        = tf.concat(output_groups, 3)

        # END IF

        if use_bias:
            b        = tf.get_variable('bias', shape=[num_channels_out], initializer=bias_init, trainable=trainable)
            conv_out = output + b

        else:
            conv_out = output

        # END IF

        if non_linear_fn is not None:
            conv_out = non_linear_fn(conv_out, name=scope.name)

        # END IF

    # END WITH

    return conv_out



def max_pool_layer(input_tensor,
                   filter_dims,
                   stride_dims,
                   name,
                   padding='SAME'):

    """
    Args:
        :input_tensor: Input tensor to the max pooling layer
        :filter_dims:  A list detailing the height and width for filters in this layer
        :stride_dims:  A list detailing the height and width of the stride between filters
        :name:         Scope name to be provided for current max pooling layer
        :padding:      Padding type definition (SAME or VALID)

    Return:
        :pool_out:     Output of max pooling layer
    """

    # Ensure parameters match required shapes
    assert(len(filter_dims) == 2)  # filter height and width
    assert(len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(name) as scope:
        # Define the max pool flow graph and return output
        pool_out = tf.nn.max_pool(input_tensor, ksize=[1, filter_h, filter_w, 1],
                               strides=[1, stride_h, stride_w, 1], padding=padding, name=scope.name)
    # END WITH

    return pool_out


def max_pool3d_layer(input_tensor,
                     filter_dims,
                     stride_dims,
                     name,
                     padding='SAME'):

    """
    Args:
        :input_tensor: Input tensor to the max pooling layer
        :filter_dims:  A list detailing the depth, height, and width for filters in this layer
        :stride_dims:  A list detailing the depth, height, and width of the stride between filters
        :name:         Scope name to be provided for current max pooling layer
        :padding:      Padding type definition (SAME or VALID)

    Return:
        :pool_out:     Output of max pooling layer
    """

    # Ensure parameters match required shapes
    assert((len(filter_dims) == 5) or (len(filter_dims) == 3))  # filter depth height and width
    assert((len(stride_dims) == 5) or (len(stride_dims) == 3))  # stride depth height and width

    if len(filter_dims) == 5:
    	_, filter_d, filter_h, filter_w, _ = filter_dims
	_, stride_d, stride_h, stride_w, _ = stride_dims

    else:
    	filter_d, filter_h, filter_w = filter_dims
	stride_d, stride_h, stride_w = stride_dims

    with tf.variable_scope(name) as scope:
        # Define the max pool flow graph and return output
        pool_out = tf.nn.max_pool3d(input_tensor, ksize=[1, filter_d, filter_h, filter_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1], padding=padding, name=scope.name)
    # END WITH

    return pool_out


def avg_pool_layer(input_tensor,
                   filter_dims,
                   stride_dims,
                   name,
                   padding='SAME'):
    """
    Args:
        :input_tensor: Input tensor to the average pooling layer
        :filter_dims:  A list detailing the height and width for filters in this layer
        :stride_dims:  A list detailing the height and width of the stride between filters
        :name:         Scope name to be provided for current max pooling layer
        :padding:      Padding type definition (SAME or VALID)

    Return:
        :pool_out:     Output of average pooling layer
    """

    # Ensure parameters match required shapes
    assert(len(filter_dims) == 2)  # filter height and width
    assert(len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims
    with tf.variable_scope(name) as scope:
        # Define the max pool flow graph and return output
        pool_out = tf.nn.avg_pool(input_tensor, ksize=[1, filter_h, filter_w, 1],
                               strides=[1, stride_h, stride_w, 1], padding=padding, name=scope.name)
    return pool_out


def avg_pool3d_layer(input_tensor,
                     filter_dims,
                     stride_dims,
                     name,
                     padding='SAME'):
    """
    Args:
        :input_tensor: Input tensor to the average pooling layer
        :filter_dims:  A list detailing the height and width for filters in this layer
        :stride_dims:  A list detailing the height and width of the stride between filters
        :name:         Scope name to be provided for current max pooling layer
        :padding:      Padding type definition (SAME or VALID)

    Return:
        :pool_out:     Output of average pooling layer
    """

    # Ensure parameters match required shapes
    assert(len(filter_dims) == 5)  # filter height and width
    assert(len(stride_dims) == 5)  # stride height and width

    _, filter_d, filter_h, filter_w, _ = filter_dims
    _, stride_d, stride_h, stride_w, _ = stride_dims
    with tf.variable_scope(name) as scope:
        # Define the max pool flow graph and return output
        pool_out = tf.nn.avg_pool3d(input_tensor, ksize=[1, filter_d, filter_h, filter_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1], padding=padding, name=scope.name)
    return pool_out


def fully_connected_layer(input_tensor,
                          out_dim,
                          name,
                          weight_decay=0.0,
                          trainable=True,
                          non_linear_fn=tf.nn.relu,
                          weight_init=tf.truncated_normal_initializer(stddev=0.01),
                          bias_init=tf.constant_initializer(0.1)):

    """
    Args:
        :input_tensor:  Input tensor to the fully connected layer
        :out_dim:       Number of output dimensions
        :name:          Scope name to be provided for current fully connected layer
        :non_linear_fn: Activation function applied to the outcome of the layer
        :weight_init:   Tensorflow initialization function used to initialize the weight matrix
        :bias_init:     Tensorflow initialization function used to initialize the bias

    Return:
        :fc_out:        Output of the fully connected layer
    """

    assert (type(out_dim) == int)

    with tf.variable_scope(name) as scope:
        input_dims = input_tensor.get_shape().as_list()

        if len(input_dims) == 4:
            batch_size, input_h, input_w, num_channels = input_dims

            in_dim     = input_h * input_w * num_channels
            flat_input = tf.reshape(input_tensor, [-1, in_dim])

        else:
            in_dim     = input_dims[-1]
            flat_input = input_tensor

        # END IF

        w      = tf.get_variable('kernel', shape=[in_dim, out_dim], initializer=weight_init, regularizer=tf.contrib.layers.l2_regularizer(weight_decay), trainable=trainable)
        b      = tf.get_variable('bias', shape=[out_dim], initializer=bias_init, trainable=trainable)
        fc_out = tf.add(tf.matmul(flat_input, w), b, name=scope.name)

        if non_linear_fn is not None:
            fc_out = non_linear_fn(fc_out)

        # END IF

    # END WITH

    return fc_out


def reshape(input_tensor, shape, name):
    """
    Args:
        :input_tensor:  Input tensor to be reshaped
        :shape:         Shape to reshape input tensor into
        :name:          Scope name to be provided for reshape operation

    Return:
        Reshaped Tensor
    """
    return tf.reshape(input_tensor, shape=shape, name=name)


def dropout(input_tensor, training, rate):
    """
    Args:
        :input_tensor:  Input tensor to be reshaped
        :training:      Whether to return output during training or testing
        :rate:          Dropout rate (ex. 0.1 would drop 10% of inputs)

    Return:
        Dropout applied to input tensorReshaped Tensor
    """
    return tf.layers.dropout(input_tensor, training=training, rate=rate)


def batch_normalization(input_tensor, training, name):
    """
    Args:
        :input_tensor:  Input tensor to be reshaped
        :training:      Whether to return output during training or testing
        :name:          Scope name to be provided for reshape operation

    Return:
        Batch normalized input tensor
    """
    return tf.layers.batch_normalization(input_tensor, training=training, name=name)


def pad(input_tensor,
        padding):

    """
    Function to return padded tensors upto a specified thickness
    Args:
        :param input_tensor: Input tensor
        :param padding: Number of padding zeros on 1 side

    Returns:
        Padded output tensor
    """
    # Pad all four sides of each frame with "padding" amount of zeros
    if len(input_tensor.shape) == 3:
        return tf.pad(input_tensor, [[padding, padding],[padding, padding],[0,0]], "CONSTANT")

    elif len(input_tensor.shape) == 4:
        return tf.pad(input_tensor, [[0,0],[padding, padding],[padding, padding],[0,0]], "CONSTANT")

    elif len(input_tensor.shape) == 5:
        return tf.pad(input_tensor, [[0,0],[0,0],[padding, padding],[padding, padding],[0,0]], "CONSTANT")

def lstm(inputs, seq_length, feat_size, cell_size=1024):
    """
    Args:
        :inputs:       List of length input_dims where each element is of shape [batch_size x feat_size]
        :seq_length:   Length of output sequence
        :feat_size:    Size of input to LSTM
        :cell_size:    Size of internal cell (output of LSTM)

    Return:
        :lstm_outputs:  Output list of length seq_length where each element is of shape [batch_size x cell_size]
    """

    # Unstack input tensor to match shape:
    # list of n_time_steps items, each item of size (batch_size x featSize)
    inputs = tf.reshape(inputs, [-1,1,feat_size])
    inputs = tf.unstack(inputs, seq_length, axis=0)

    # LSTM cell definition
    lstm_cell            = tf.contrib.rnn.BasicLSTMCell(cell_size)
    lstm_outputs, states = static_rnn(lstm_cell, inputs, dtype=tf.float32)

    # Condense output shape from:
    # list of n_time_steps itmes, each item of size (batch_size x cell_size)
    # To:
    # Tensor: [(n_time_steps x 1), cell_size] (Specific to our case)
    lstm_outputs = tf.stack(lstm_outputs)
    lstm_outputs = tf.reshape(lstm_outputs,[-1,cell_size])

    return lstm_outputs
