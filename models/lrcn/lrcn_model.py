" LRCN MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "

import os
import sys
sys.path.append('../..')

import tensorflow as tf
import numpy      as np

from rnn_cell_impl                import LSTMCell

#from lrcn_preprocessing           import preprocess
from lrcn_preprocessing_TFRecords import preprocess as preprocess_tfrecords
from layers_utils                 import *

class LRCN():

    def __init__(self, verbose=True):
        """
        Args:
            :verbose: Setting verbose command
        """
        self.verbose=verbose
        self.name = 'lrcn'
        print "lrcn initialized"


    def _LSTM(self, inputs, is_training, input_dims, seq_length, data_dict, feat_size=4096, cell_size=256, gpu_id=0):
        """
        Args:
            :inputs:       List of length input_dims where each element is of shape [batch_size x feat_size]
            :is_training:  Boolean variable indicating phase (TRAINING OR TESTING)
            :input_dims:   Length of input sequence
            :seq_length:   Length of output sequence
            :data_dict:    Data dictionary to return value of LSTM from Lisa's model
            :feat_size:    Size of input to LSTM
            :cell_size:    Size of internal cell (output of LSTM)
            :gpu_id:       GPU ID for current LSTM scope

        Return:
            :rnn_outputs:  Output list of length seq_length where each element is of shape [batch_size x cell_size]
        """


        inputs = tf.reshape(inputs, shape=[input_dims/seq_length,seq_length,feat_size])

        wi = tf.get_variable('rnn/lstm_cell/kernel', [4352, 1024], initializer=tf.constant_initializer(data_dict['lstm1'][0]))
        bi = tf.get_variable('rnn/lstm_cell/bias',   [1024],       initializer=tf.constant_initializer(data_dict['lstm1'][1]))

        lstm_cell           = LSTMCell(cell_size, forget_bias=0.0, weights_initializer=wi, bias_initializer=bi)
        rnn_outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
        rnn_outputs         = tf.reshape(rnn_outputs, shape=[-1,256])



        return rnn_outputs

    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, K, J, dropout_rate = 0.5, return_layer='logits', gpu_id = 0, weight_decay=0.0):
        """
        Args:
            :inputs:       Input to model of shape [Frames x Height x Width x Channels]
            :is_training:  Boolean variable indicating phase (TRAIN OR TEST)
            :input_dims:   Length of input sequence
            :output_dims:  Integer indicating total number of classes in final prediction
            :seq_length:   Length of output sequence from LSTM
            :scope:        Scope name for current model instance
            :dropout_rate: Value indicating proability of keep inputs
            :return_layer: String matching name of a layer in current model
            :gpu_id:       GPU ID for current model
            :weight_decay: Double value of weight decay

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """

        ############################################################################
        #                        Creating LRCN Network Layers                      #
        ############################################################################


        if self.verbose:
            print('Generating LRCN Network Layers')

        # END IF

        # Must exist within current model directory
        data_dict = np.load('models/lrcn/lrcn.npy').tolist()

        with tf.name_scope(scope, 'lrcn', [inputs]):
            layers = {}

            layers['conv1'] = conv_layer(input_tensor=inputs,
                           filter_dims=[7,7,96],
                           stride_dims=[2,2],
                           name='conv1',
                           weight_decay=weight_decay,
                           padding='VALID',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(data_dict['conv1'][0]),
                           bias_init=tf.constant_initializer(data_dict['conv1'][1]))

            layers['pool1'] = max_pool_layer(layers['conv1'],
                               filter_dims=[3,3],
                               stride_dims=[2,2],
                               name='pool1',
                               padding='VALID')

            layers['norm1'] = tf.nn.lrn(input=layers['pool1'], depth_radius=2, alpha=0.0001/5, beta=0.75, name='norm1')

            layers['conv2'] = conv_layer(input_tensor=layers['norm1'],
                           filter_dims=[5,5,384],
                           stride_dims=[2,2],
                           name='conv2',
                           weight_decay=weight_decay,
                           padding='VALID',
                           groups=2,
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(data_dict['conv2'][0]),
                           bias_init=tf.constant_initializer(data_dict['conv2'][1]))

            layers['pool2'] = max_pool_layer(layers['conv2'],
                               filter_dims=[3,3],
                               stride_dims=[2,2],
                               name='pool2',
                               padding='SAME')

            layers['norm2'] = tf.nn.lrn(input=layers['pool2'], depth_radius=2, alpha=0.0001/5, beta=0.75, name='norm2')

            layers['conv3'] = conv_layer(input_tensor=layers['norm2'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv3',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(data_dict['conv3'][0]),
                           bias_init=tf.constant_initializer(data_dict['conv3'][1]))

            layers['conv4'] = conv_layer(input_tensor=layers['conv3'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv4',
                           weight_decay=weight_decay,
                           padding='SAME',
                           groups=2,
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(data_dict['conv4'][0]),
                           bias_init=tf.constant_initializer(data_dict['conv4'][1]))

            layers['conv5'] = conv_layer(input_tensor=layers['conv4'],
                           filter_dims=[3,3,384],
                           stride_dims=[1,1],
                           name='conv5',
                           weight_decay=weight_decay,
                           padding='SAME',
                           groups=2,
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(data_dict['conv5'][0]),
                           bias_init=tf.constant_initializer(data_dict['conv5'][1]))

            layers['pool5'] = max_pool_layer(layers['conv5'],
                               filter_dims=[3,3],
                               stride_dims=[2,2],
                               name='pool5',
                               padding='VALID')

            layers['fc6'] = fully_connected_layer(input_tensor=layers['pool5'],
                                      out_dim=4096,
                                      name='fc6',
                                      weight_decay=weight_decay,
                                      non_linear_fn=tf.nn.relu,
                                      weight_init=tf.constant_initializer(data_dict['fc6'][0]),
                                      bias_init=tf.constant_initializer(data_dict['fc6'][1]))

            layers['rnn_outputs_rs'] = self._LSTM(layers['fc6'], is_training, input_dims, seq_length, data_dict, cell_size=256)


            layers['logits'] = fully_connected_layer(input_tensor=layers['rnn_outputs_rs'],
                                      out_dim=output_dims,
                                      name='logits',
                                      weight_decay=weight_decay,
                                      non_linear_fn=None,
                                      weight_init=tf.constant_initializer(data_dict['fc8'][0]),
                                      bias_init=tf.constant_initializer(data_dict['fc8'][1]))

            # END WITH

        return layers[return_layer]

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


    # def preprocess(self, index, data, labels, size, is_training):
    #     """
    #     Args:
    #         :index:       Integer indicating the index of video frame from the text file containing video lists
    #         :data:        Data loaded from HDF5 files
    #         :labels:      Labels for loaded data
    #         :size:        List detailing values of height and width for final frames
    #         :is_training: Boolean value indication phase (TRAIN OR TEST)
    #     """
    #     return preprocess(index, data,labels, size, is_training)

    """ Function to return loss calculated on given network """
    def loss(self, logits, labels):
        """
        Args:
            :logits: Unscaled logits returned from final layer in model
            :labels: True labels corresponding to loaded data
        """

        labels = tf.tile(labels, logits.shape[0].value/labels.shape[0].value)
        labels = tf.cast(labels, tf.int64)

        crossEntropyLoss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                  logits=logits)

        return crossEntropyLoss
