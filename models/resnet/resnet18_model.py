" RESNET-18 + LSTM MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "

import h5py
import os
import time
import sys
sys.path.append('../..')

import tensorflow as tf
import numpy      as np

from tensorflow.contrib.rnn         import static_rnn
from layers_utils                   import *
from resnet_preprocessing_TFRecords import preprocess   as preprocess_tfrecords

class ResNet18():

    def __init__(self, verbose=True):
        """
        Args:
            :verbose: Setting verbose command
        """
        self.verbose=verbose
        self.name = 'resnet18'
        print "resnet18 initialized"

    def _LSTM(self, inputs, seq_length, feat_size, cell_size=1024):
        """
        Args:
            :inputs:       List of length input_dims where each element is of shape [batch_size x feat_size]
            :seq_length:   Length of output sequence
            :feat_size:    Size of input to LSTM
            :cell_size:    Size of internal cell (output of LSTM)

        Return:
            :lstn_outputs:  Output list of length seq_length where each element is of shape [batch_size x cell_size]
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

    def _conv_block(self, n_filters, kernel_size, name, layer_numbers, input_layer, data_dict, strides=2, weight_decay=0.0):
        """
        Args:
            :n_filters:     List detailing the number of filters
            :kernel_size:   List detailing the height and width of the kernel
            :name:          Name of the conv_block branch
            :layer_numbers: List detailing the connecting layer indices
            :input_layer:   Input layer to the conv_block
            :data_dict:     Data dictionary containing initializers
            :strides:       Integer value for stride between filters
            :weight_decay:  Double value of weight decay

        Return:
            :layers:        Stack of layers
        """

        layers = {}

        # Conv block
        layers[layer_numbers[0]] = conv_layer(input_tensor=input_layer,
                filter_dims=[kernel_size,kernel_size,n_filters[0]], stride_dims=[strides,strides],
                padding = 'SAME',
                name=name+'/conv_1/',
                kernel_init=tf.constant_initializer(data_dict[name+'/conv_1/kernel']),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[1]] = tf.layers.batch_normalization(layers[layer_numbers[0]],
                name=name+'/bn_1/',
                moving_mean_initializer=tf.constant_initializer(data_dict[name+'/bn_1/mu']),
                moving_variance_initializer=tf.constant_initializer(data_dict[name+'/bn_1/sigma']),
                beta_initializer=tf.constant_initializer(data_dict[name+'/bn_1/beta']),
                gamma_initializer=tf.constant_initializer(data_dict[name+'/bn_1/gamma']))

        layers[layer_numbers[2]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[1]]),
                filter_dims=[kernel_size,kernel_size,n_filters[1]], stride_dims=[strides,strides],
                padding = 'SAME',
                name=name+'/conv_2/',
                kernel_init=tf.constant_initializer(data_dict[name+'/conv_2/kernel']),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[3]] = tf.layers.batch_normalization(layers[layer_numbers[2]],
                name=name+'/bn_2/',
                moving_mean_initializer=tf.constant_initializer(data_dict[name+'/bn_2/mu']),
                moving_variance_initializer=tf.constant_initializer(data_dict[name+'/bn_2/sigma']),
                beta_initializer=tf.constant_initializer(data_dict[name+'/bn_2/beta']),
                gamma_initializer=tf.constant_initializer(data_dict[name+'/bn_2/gamma']))

        # END OF CONV BLOCK

        layers[layer_numbers[4]] = tf.nn.relu(tf.add(input_layer,layers[layer_numbers[3]]))

        return layers


    def _identity_block(self, n_filters, kernel_size, name, layer_numbers, input_layer, data_dict, strides=2, weight_decay=0.0):
        """
        Args:
            :n_filters:     List detailing the number of filters
            :kernel_size:   List detailing the height and width of the kernel
            :name:          Name of the identity_block branch
            :layer_numbers: List detailing the connecting layer indices
            :input_layer:   Input layer to the identity_block
            :data_dict:     Data dictionary containing initializers
            :strides:       Integer value for stride between filters
            :weight_decay:  Double value of weight decay

        Return:
            :layers:        Stack of layers
        """

        layers = {}

        # Identity block
        layers[layer_numbers[0]] = conv_layer(input_tensor=input_layer,
                filter_dims=[kernel_size,kernel_size,n_filters[0]], stride_dims=[strides,strides],
                padding = 'SAME',
                name=name+'/conv_1/',
                kernel_init=tf.constant_initializer(data_dict[name+'/conv_1/kernel']),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[1]] = tf.layers.batch_normalization(layers[layer_numbers[0]],
                name=name+'/bn_1/',
                moving_mean_initializer=tf.constant_initializer(data_dict[name+'/bn_1/mu']),
                moving_variance_initializer=tf.constant_initializer(data_dict[name+'/bn_1/sigma']),
                beta_initializer=tf.constant_initializer(data_dict[name+'/bn_1/beta']),
                gamma_initializer=tf.constant_initializer(data_dict[name+'/bn_1/gamma']))

        layers[layer_numbers[2]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[1]]),
                filter_dims=[kernel_size,kernel_size,n_filters[1]], stride_dims=[1,1],
                padding = 'SAME',
                name=name+'/conv_2/',
                kernel_init=tf.constant_initializer(data_dict[name+'/conv_2/kernel']),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[3]] = tf.layers.batch_normalization(layers[layer_numbers[2]],
                name=name+'/bn_2/',
                moving_mean_initializer=tf.constant_initializer(data_dict[name+'/bn_2/mu']),
                moving_variance_initializer=tf.constant_initializer(data_dict[name+'/bn_2/sigma']),
                beta_initializer=tf.constant_initializer(data_dict[name+'/bn_2/beta']),
                gamma_initializer=tf.constant_initializer(data_dict[name+'/bn_2/gamma']))

        # Shortcut
        layers[layer_numbers[4]] = conv_layer(input_tensor=input_layer,
                filter_dims=[1,1,n_filters[2]], stride_dims=[strides,strides],
                padding = 'SAME',
                name=name+'/shortcut/',
                kernel_init=tf.constant_initializer(data_dict[name+'/shortcut/kernel']),
                weight_decay = weight_decay, non_linear_fn=None)

        # END OF IDENTITY BLOCK

        layers[layer_numbers[5]] = tf.nn.relu(tf.add(layers[layer_numbers[4]], layers[layer_numbers[3]]))

        return [layers[x] for x in return_layer]


    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, k, j, dropout_rate = 0.5, return_layer=['logits'], data_dict=None, weight_decay=0.0):
        """
        Args:
            :inputs:       Input to model of shape [Frames x Height x Width x Channels]
            :is_training:  Boolean variable indicating phase (TRAIN OR TEST)
            :input_dims:   Length of input sequence
            :output_dims:  Integer indicating total number of classes in final prediction
            :seq_length:   Length of output sequence from LSTM
            :scope:        Scope name for current model instance
            :k:            Width of sliding window (temporal width)
            :j:            Integer number of disjoint sets the sliding window over the input has generated
            :dropout_rate: Value indicating proability of keep inputs
            :return_layer: String matching name of a layer in current model
            :weight_decay: Double value of weight decay

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """

        ############################################################################
        #                Creating ResNet18 + LSTM Network Layers                   #
        ############################################################################

        if self.verbose:
            print('Generating RESNET network layers')

        # END IF

        # Must exist within current model directory
        #data_dict = np.load('models/resnet/resnet18_weights_tf_dim_ordering_tf_kernels.npy')
        data_dict = np.load('models/resnet/resnet18_weights_tf_dim_ordering_tf_kernels.npy').item()

        with tf.name_scope(scope, 'resnet', [inputs]):
            layers = {}

            layers['1'] = conv_layer(input_tensor=inputs,
                    filter_dims=[7, 7, 64], stride_dims=[2,2],
                    padding = 'VALID',
                    name='conv1',
                    kernel_init=tf.constant_initializer(data_dict['conv1/conv/kernel']),
                    weight_decay = weight_decay, non_linear_fn=None)

            layers['2'] = tf.layers.batch_normalization(layers['1'],
                    name='bn_conv1',
                    moving_mean_initializer=tf.constant_initializer(data_dict['conv1/bn/mu']),
                    moving_variance_initializer=tf.constant_initializer(data_dict['conv1/bn/sigma']),
                    beta_initializer=tf.constant_initializer(data_dict['conv1/bn/beta']),
                    gamma_initializer=tf.constant_initializer(data_dict['conv1/bn/gamma']))

            layers['3'] = max_pool_layer(tf.nn.relu(layers['2']),
                                         filter_dims=[3, 3], stride_dims=[2,2],
                                         name='pool1', padding='SAME')

            layers.update(self._conv_block([64,64], kernel_size=3, name='conv2_1', layer_numbers=['4','5','6','7','8'],
                            input_layer=layers['3'], strides=1, data_dict=data_dict))

            layers.update(self._conv_block([64,64], kernel_size=3, name='conv2_2', layer_numbers=['9','10','11','12','13'],
                            input_layer=layers['8'], strides=1, data_dict=data_dict))

            layers.update(self._identity_block([128,128,128], kernel_size=3, name='conv3_1', layer_numbers=['14','15','16','17','18','19'],
                            input_layer=layers['13'], data_dict=data_dict))

            layers.update(self._conv_block([128,128], kernel_size=3, name='conv3_2', layer_numbers=['20','21','22','23','24'],
                            input_layer=layers['19'], strides=1, data_dict=data_dict))

            layers.update(self._identity_block([256,256,256], kernel_size=3, name='conv4_1', layer_numbers=['25','26','27','28','29','30'],
                            input_layer=layers['24'], data_dict=data_dict))

            layers.update(self._conv_block([256,256], kernel_size=3, name='conv4_2', layer_numbers=['31','32','33','34','35'],
                            input_layer=layers['30'], strides=1, data_dict=data_dict))

            layers.update(self._identity_block([512,512,512], kernel_size=3, name='conv5_1', layer_numbers=['36','37','38','39','40','41'],
                            input_layer=layers['35'], data_dict=data_dict))

            layers.update(self._conv_block([512,512], kernel_size=3, name='conv5_2', layer_numbers=['42','43','44','45','46'],
                            input_layer=layers['41'], strides=1, data_dict=data_dict))

            layers['47'] = tf.reduce_mean(layers['46'], reduction_indices=[1,2], name='avg_pool')
            layers['48'] = self._LSTM(layers['47'], seq_length, feat_size=512, cell_size=512)

            layers['49'] = tf.layers.dropout(layers['48'], training=is_training, rate=0.5)

            layers['logits'] = fully_connected_layer(input_tensor=layers['49'],
                                                     out_dim=output_dims, non_linear_fn=None,
                                                     name='logits', weight_decay=weight_decay)
            # END WITH

        return layers[return_layer]


    def load_default_weights(self):
        """
        return: Numpy dictionary containing the names and values of the weight tensors used to initialize this model
        """
        return np.load('models/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.npy')


    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining):
        """
        Args:
            :index:       Integer indicating the index of video frame from the text file containing video lists
            :data:        Data loaded from HDF5 files
            :labels:      Labels for loaded data
            :size:        List detailing values of height and width for final frames
            :is_training: Boolean value indication phase (TRAIN OR TEST)
        """
        return preprocess_tfrecords(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining)

    """ Function to return loss calculated on given network """
    def loss(self, logits, labels):
        """
        Args:
            :logits: Unscaled logits returned from final layer in model
            :labels: True labels corresponding to loaded data
        """

        labels = tf.cast(labels, tf.int64)

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels[:labels.shape[0].value/2],
                                                                  logits=logits[:logits.shape[0].value/2,:])
        return cross_entropy_loss




if __name__ == '__main__':
    XX = ResNet18()
    llol = XX.inference(tf.placeholder(tf.float32, shape=[50,224,224,3]), True, 50, 51, 50, 'mine', 25, 10, dropout_rate = 0.5, return_layer='logits', data_dict=None, weight_decay=0.0)
