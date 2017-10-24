import tensorflow as tf
import numpy as np


def conv_layer(input_tensor,
               filter_dims,
               name,
               stride_dims = [1,1],
               weight_decay=0.0,
               padding='SAME',
               groups=1,
               non_linear_fn=tf.nn.relu,
               kernel_init=tf.truncated_normal_initializer(stddev=0.01),
               bias_init=tf.constant_initializer(0.1)):
    """
    :param input_tensor:
    :param filter_dims:
    :param stride_dims:
    :param name:
    :param padding:
    :param non_linear_fn:
    :param kernel_init:
    :param bias_init:
    :return:
    """
    input_dims = input_tensor.get_shape().as_list()
    assert (len(input_dims) == 4)
    assert(len(filter_dims) == 3)
    assert(len(stride_dims) == 2)

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride_h, stride_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        if groups == 1:
            w = tf.get_variable('kernel', shape=[filter_h, filter_w, num_channels_in, num_channels_out],
                                initializer=kernel_init, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            output = convolve(input_tensor, w)
        else:
            w = tf.get_variable('kernel', shape=[filter_h, filter_w, int(num_channels_in/groups), num_channels_out],
                                initializer=kernel_init, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            input_groups = tf.split(input_tensor, groups, axis=3)
            kernel_groups = tf.split(w, groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            output = tf.concat(output_groups, 3)
        b = tf.get_variable('bias', shape=[num_channels_out], initializer=bias_init)
        conv_out = output + b
        if non_linear_fn is not None:
            conv_out = non_linear_fn(conv_out, name=scope.name)
    return conv_out


def max_pool_layer(input_tensor,
                   filter_dims,
                   stride_dims,
                   name,
                   padding='SAME'):
    """
    :param input_tensor:
    :param filter_dims:
    :param stride_dims:
    :param name:
    :param padding:
    :return:
    """
    assert(len(filter_dims) == 2)  # filter height and width
    assert(len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims
    with tf.variable_scope(name) as scope:
        # Define the max pool flow graph and return output
        pool1 = tf.nn.max_pool(input_tensor, ksize=[1, filter_h, filter_w, 1],
                               strides=[1, stride_h, stride_w, 1], padding=padding, name=scope.name)
    return pool1


def fully_connected_layer(input_tensor,
                          out_dim,
                          name,
                          weight_decay=0.0,
                          non_linear_fn=tf.nn.relu,
                          weight_init=tf.truncated_normal_initializer(stddev=0.01),
                          bias_init=tf.constant_initializer(0.1)):
    """
    :param input_tensor:
    :param out_dim:
    :param name:
    :param non_linear_fn:
    :param weight_init:
    :param bias_init:
    :return:
    """
    assert (type(out_dim) == int)
    with tf.variable_scope(name) as scope:
        input_dims = input_tensor.get_shape().as_list()
        if len(input_dims) == 4:
            batch_size, input_h, input_w, num_channels = input_dims
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input_tensor, [-1, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input_tensor

        w = tf.get_variable('weights', shape=[in_dim, out_dim], initializer=weight_init, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        b = tf.get_variable('bias', shape=[out_dim], initializer=bias_init)
        fc1 = tf.add(tf.matmul(flat_input, w), b, name=scope.name)
        if non_linear_fn is not None:
            fc1 = non_linear_fn(fc1)
        return fc1
#
# def batch_normalization_layer(input_tensor,
#                               name,
#                               moving_mean_initializer=tf.zeros_initializer(),
#                               moving_variance_initializer=tf.ones_initializer(),
#                               beta_initializer=tf.zeros_initializer(),
#                               gamma_initializer=tf.ones_initializer()):
#     """
#     :param input_tensor:
#     :param out_dim:
#     :param name:
#     :param non_linear_fn:
#     :param weight_init:
#     :param bias_init:
#     :return:
#     """
#     with tf.variable_scope(name) as scope:
#
