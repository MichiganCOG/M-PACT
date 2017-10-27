# LRCN Model implemnetation for use with tensorflow

import os
import tensorflow as tf
import numpy as np
from rnn_cell_impl import LSTMCell
from lrcn_preprocessing import preprocess

import sys
sys.path.append('../..')
from layers_utils import *

class LRCN():

    def __init__(self, verbose=True):
        self.verbose=verbose
        self.name = 'lrcn'


    def _LSTM(self, inputs, isTraining, inputDims, seqLength, dataDict, featSize=4096, cellSize=256, cpuId=0):

        with tf.device('/gpu:'+str(cpuId)):

            inputs = tf.reshape(inputs, shape=[inputDims/seqLength,seqLength,featSize])

            wi = tf.get_variable('rnn/lstm_cell/kernel', [4352, 1024], initializer=tf.constant_initializer(dataDict['lstm1'][0]))
            bi = tf.get_variable('rnn/lstm_cell/bias', [1024], initializer=tf.constant_initializer(dataDict['lstm1'][1]))
            lstm_cell = LSTMCell(cellSize, forget_bias=0.0, weights_initializer=wi, bias_initializer=bi)
            rnn_outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
            rnn_outputs = tf.reshape(rnn_outputs, shape=[-1,256])
            return rnn_outputs
    def inference(self, inputs, isTraining, inputDims, outputDims, seqLength, scope, dropoutRate = 0.5, return_layer='logits', cpuId = 0, weight_decay=0.0):

        ############################################################################
        #                        Creating LRCN Network Layers                      #
        ############################################################################


        if self.verbose:
            print('Generating LRCN Network Layers')


        dataDict = np.load('models/lrcn/lrcn.npy').tolist()

        with tf.name_scope("my_scope"):

            layers = {}

            layers['conv1'] = conv_layer(input_tensor=inputs,
                           filter_dims=[7,7,96],
                           stride_dims=[2,2],
                           name='conv1',
                           weight_decay=weight_decay,
                           padding='VALID',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv1'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv1'][1]))



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
                           kernel_init=tf.constant_initializer(dataDict['conv2'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv2'][1]))



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
                           kernel_init=tf.constant_initializer(dataDict['conv3'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv3'][1]))




            layers['conv4'] = conv_layer(input_tensor=layers['conv3'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv4',
                           weight_decay=weight_decay,
                           padding='SAME',
                           groups=2,
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv4'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv4'][1]))




            layers['conv5'] = conv_layer(input_tensor=layers['conv4'],
                           filter_dims=[3,3,384],
                           stride_dims=[1,1],
                           name='conv5',
                           weight_decay=weight_decay,
                           padding='SAME',
                           groups=2,
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv5'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv5'][1]))




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
                                      weight_init=tf.constant_initializer(dataDict['fc6'][0]),
                                      bias_init=tf.constant_initializer(dataDict['fc6'][1]))



            layers['rnn_outputs_rs'] = self._LSTM(layers['fc6'], isTraining, inputDims, seqLength, dataDict, cellSize=256)


            layers['logits'] = fully_connected_layer(input_tensor=layers['rnn_outputs_rs'],
                                      out_dim=outputDims,
                                      name='logits',
                                      weight_decay=weight_decay,
                                      non_linear_fn=None,
                                      weight_init=tf.constant_initializer(dataDict['fc8'][0]),
                                      bias_init=tf.constant_initializer(dataDict['fc8'][1]))

            return layers[return_layer]







    def preprocess(self, index, data, labels, size, isTraining):
        return preprocess(index, data,labels, size, isTraining)


    """ Function to return loss calculated on given network """
    def loss(self, logits, labels):

        labels = tf.cast(labels, tf.int64)
        crossEntropyLoss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                        logits=logits)

        return crossEntropyLoss
