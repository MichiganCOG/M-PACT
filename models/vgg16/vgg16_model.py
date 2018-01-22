" VGG16 + LSTM MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "

import os

import tensorflow as tf
import numpy      as np

from utils.layers_utils            import *
from tensorflow.contrib.rnn        import static_rnn
from vgg16_preprocessing_TFRecords import preprocess   as preprocess_tfrecords



class VGG16():

    def __init__(self, input_dims, k, verbose=True):
        """
        Args:
            :k:          Temporal window width
            :verbose:    Setting verbose command
            :input_dims: Input dimensions (number of frames)

        Return:
            Does not return anything
        """
        self.k          = k
        self.name       = 'vgg16'
        self.verbose    = verbose
        self.input_dims = input_dims
        self.j          = input_dims / k

        print "VGG16 + LSTM initialized"

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

        inputs = tf.unstack(inputs, seq_length, axis=0)

        # LSTM cell definition
        lstm_cell            = tf.contrib.rnn.BasicLSTMCell(cell_size)
        lstm_outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, inputs, dtype=tf.float32)

        # Condense output shape from:
        # list of n_time_steps itmes, each item of size (batch_size x cell_size)
        # To:
        # Tensor: [(n_time_steps x 1), cell_size] (Specific to our case)
        lstm_outputs = tf.stack(lstm_outputs)
        lstm_outputs = tf.reshape(lstm_outputs,[-1,cell_size])

        return lstm_outputs


    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, weight_decay=0.0, return_layer=['logits']):

        """
        Args:
            :inputs:       Input to model of shape [Frames x Height x Width x Channels]
            :is_training:  Boolean variable indicating phase (TRAIN OR TEST)
            :input_dims:   Length of input sequence
            :output_dims:  Integer indicating total number of classes in final prediction
            :seq_length:   Length of output sequence from LSTM
            :scope:        Scope name for current model instance
            :return_layer: String matching name of a layer in current model
            :weight_decay: Double value of weight decay

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """

        ############################################################################
        #                       Creating VGG 16 Network Layers                     #
        ############################################################################

        if self.verbose:
            print('Generating VGG16 network layers')

        # END IF

        inputs = inputs[0]

        if is_training:
            keep_prob = 0.5
        else:
            keep_prob = 1.0

        # END IF

        with tf.name_scope(scope, 'vgg16', [inputs]):
            layers = {}

            layers['conv1'] = conv_layer(input_tensor=inputs,
                           filter_dims=[3,3,64],
                           stride_dims=[1,1],
                           name='conv1',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['conv2'] = conv_layer(input_tensor=layers['conv1'],
                           filter_dims=[3,3,64],
                           stride_dims=[1,1],
                           name='conv2',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['pool2'] = max_pool_layer(layers['conv2'],
                               filter_dims=[2,2],
                               stride_dims=[2,2],
                               name='pool2',
                               padding='VALID')

            layers['conv3'] = conv_layer(input_tensor=layers['pool2'],
                           filter_dims=[3,3,128],
                           stride_dims=[1,1],
                           name='conv3',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['conv4'] = conv_layer(input_tensor=layers['conv3'],
                           filter_dims=[3,3,128],
                           stride_dims=[1,1],
                           name='conv4',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['pool4'] = max_pool_layer(layers['conv4'],
                               filter_dims=[2,2],
                               stride_dims=[2,2],
                               name='pool4',
                               padding='VALID')

            layers['conv5'] = conv_layer(input_tensor=layers['pool4'],
                           filter_dims=[3,3,256],
                           stride_dims=[1,1],
                           name='conv5',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['conv6'] = conv_layer(input_tensor=layers['conv5'],
                           filter_dims=[3,3,256],
                           stride_dims=[1,1],
                           name='conv6',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['conv7'] = conv_layer(input_tensor=layers['conv6'],
                           filter_dims=[3,3,256],
                           stride_dims=[1,1],
                           name='conv7',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['pool7'] =  max_pool_layer(layers['conv7'],
                               filter_dims=[2,2],
                               stride_dims=[2,2],
                               name='pool7',
                               padding='VALID')

            layers['conv8'] = conv_layer(input_tensor=layers['pool7'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv8',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['conv9'] = conv_layer(input_tensor=layers['conv8'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv9',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['conv10'] = conv_layer(input_tensor=layers['conv9'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv10',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['pool10'] = max_pool_layer(layers['conv10'],
                               filter_dims=[2,2],
                               stride_dims=[2,2],
                               name='pool7',
                               padding='VALID')

            layers['conv11'] = conv_layer(input_tensor=layers['pool10'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv11',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['conv12'] = conv_layer(input_tensor=layers['conv11'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv12',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['conv13'] = conv_layer(input_tensor=layers['conv12'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv13',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu)

            layers['pool13'] = max_pool_layer(layers['conv13'],
                               filter_dims=[2,2],
                               stride_dims=[2,2],
                               name='pool13',
                               padding='VALID')

            layers['fc6'] = fully_connected_layer(input_tensor=layers['pool13'],
                                      out_dim=4096,
                                      name='fc6',
                                      weight_decay=weight_decay,
                                      non_linear_fn=tf.nn.relu)

            layers['drop6'] = tf.nn.dropout(layers['fc6'], keep_prob=keep_prob)

            layers['fc7'] = fully_connected_layer(input_tensor=layers['drop6'],
                                      out_dim=4096,
                                      name='fc7',
                                      weight_decay=weight_decay,
                                      non_linear_fn=tf.nn.relu)

            layers['drop7'] = tf.nn.dropout(layers['fc7'], keep_prob=keep_prob)

            feat_size=4096
            rnn_inputs = tf.reshape(layers['drop7'], [seq_length, input_dims/seq_length,  4096])

            layers['rnn_outputs'] = self._LSTM(rnn_inputs, seq_length, feat_size=4096, cell_size=1024)



            layers['drop8'] = tf.nn.dropout(layers['rnn_outputs'], keep_prob=keep_prob)

            layers['logits'] = [fully_connected_layer(input_tensor=layers['drop8'],
                                          out_dim=output_dims,
                                          name='logits',
                                          weight_decay=weight_decay,
                                          non_linear_fn=None)]

            # END IF

        # END WITH

        return [layers[x] for x in return_layer]

    def load_default_weights(self):
        """
        return: Numpy dictionary containing the names and values of the weight tensors used to initialize this model
        """
        return np.load('models/vgg16/vgg16_new.npy')

    """ Function to return loss calculated on given network """
    def loss(self, logits, labels):
        """
        Args:
            :logits: Unscaled logits returned from final layer in model
            :labels: True labels corresponding to loaded data

        Return:
            Cross entropy loss value
        """

        labels = tf.cast(labels, tf.int64)

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels[:labels.shape[0].value/2],
                                                                    logits=logits[:logits.shape[0].value/2,:])

        return cross_entropy_loss


    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining):
        """
        Args:
            :index:       Integer indicating the index of video frame from the text file containing video lists
            :data:        Data loaded from HDF5 files
            :labels:      Labels for loaded data
            :size:        List detailing values of height and width for final frames
            :is_training: Boolean value indication phase (TRAIN OR TEST)

        Return:
            Pointer to preprocessing function of current model
        """
        return preprocess_tfrecords(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining)
