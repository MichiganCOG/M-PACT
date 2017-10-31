""" THIS FILE SUPPORTS THE GENERATION OF NETWORK SKELETONS """

# Import tensorflow modules
import tensorflow as tf

# Import math and numpy modules
import numpy as np
from vgg16_preprocessing import preprocess

import os
import sys
sys.path.append('../..')
from layers_utils import *



class VGG16():

    def __init__(self, verbose=True):
        self.verbose=verbose
        self.name = 'vgg16'

    def _LSTM(self, inputs, seqLength, featSize, weight_decay=0.0, cellSize=1024, cpuId=0):

        # Unstack input tensor to match shape:
        # list of n_time_steps items, each item of size (batch_size x featSize)


        with tf.device('/gpu:'+str(cpuId)):
            inputs = tf.unstack(inputs, seqLength, axis=0)

            # LSTM cell definition

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(cellSize)

            outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, inputs, dtype=tf.float32)

            # Condense output shape from:
            # list of n_time_steps itmes, each item of size (batch_size x cellSize)
            # To:
            # Tensor: [(n_time_steps x 1), cellSize] (Specific to our case)
            outputs = tf.stack(outputs)
            outputs = tf.reshape(outputs,[-1,cellSize])


            return outputs


    def inference(self, inputs, isTraining, inputDims, outputDims, seqLength, scope, weight_decay=0.0, return_layer='logits', cpuId=0):

        ############################################################################
        #                       Creating VGG 16 Network Layers                     #
        ############################################################################

        print('Generating VGG16 network layers')

        dataDict = np.load('/z/home/madantrg/RILCode/Code_ND/Utils/vgg16.npy').item()


        if isTraining:
            keep_prob = 0.5
        else:
            keep_prob = 1.0

        with tf.name_scope(scope, 'vgg16', [inputs]):
            layers = {}


            layers['conv1'] = conv_layer(input_tensor=inputs,
                           filter_dims=[3,3,64],
                           stride_dims=[1,1],
                           name='conv1',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv1_1'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv1_1'][1]))


            layers['conv2'] = conv_layer(input_tensor=layers['conv1'],
                           filter_dims=[3,3,64],
                           stride_dims=[1,1],
                           name='conv2',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv1_2'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv1_2'][1]))


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
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv2_1'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv2_1'][1]))



            layers['conv4'] = conv_layer(input_tensor=layers['conv3'],
                           filter_dims=[3,3,128],
                           stride_dims=[1,1],
                           name='conv4',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv2_2'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv2_2'][1]))



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
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv3_1'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv3_1'][1]))



            layers['conv6'] = conv_layer(input_tensor=layers['conv5'],
                           filter_dims=[3,3,256],
                           stride_dims=[1,1],
                           name='conv6',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv3_2'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv3_2'][1]))



            layers['conv7'] = conv_layer(input_tensor=layers['conv6'],
                           filter_dims=[3,3,256],
                           stride_dims=[1,1],
                           name='conv7',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv3_3'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv3_3'][1]))




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
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv4_1'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv4_1'][1]))



            layers['conv9'] = conv_layer(input_tensor=layers['conv8'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv9',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv4_2'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv4_2'][1]))



            layers['conv10'] = conv_layer(input_tensor=layers['conv9'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv10',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv4_3'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv4_3'][1]))



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
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv5_1'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv5_1'][1]))



            layers['conv12'] = conv_layer(input_tensor=layers['conv11'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv12',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv5_2'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv5_2'][1]))



            layers['conv13'] = conv_layer(input_tensor=layers['conv12'],
                           filter_dims=[3,3,512],
                           stride_dims=[1,1],
                           name='conv13',
                           weight_decay=weight_decay,
                           padding='SAME',
                           non_linear_fn=tf.nn.relu,
                           kernel_init=tf.constant_initializer(dataDict['conv5_3'][0]),
                           bias_init=tf.constant_initializer(dataDict['conv5_3'][1]))


            layers['pool13'] = max_pool_layer(layers['conv13'],
                               filter_dims=[2,2],
                               stride_dims=[2,2],
                               name='pool13',
                               padding='VALID')



            layers['fc6'] = fully_connected_layer(input_tensor=layers['pool13'],
                                      out_dim=4096,
                                      name='fc6',
                                      weight_decay=weight_decay,
                                      non_linear_fn=tf.nn.relu,
                                      weight_init=tf.constant_initializer(dataDict['fc6'][0]),
                                      bias_init=tf.constant_initializer(dataDict['fc6'][1]))



            layers['drop6'] = tf.nn.dropout(layers['fc6'], keep_prob=keep_prob)




            layers['fc7'] = fully_connected_layer(input_tensor=layers['drop6'],
                                      out_dim=4096,
                                      name='fc7',
                                      weight_decay=weight_decay,
                                      non_linear_fn=tf.nn.relu,
                                      weight_init=tf.constant_initializer(dataDict['fc7'][0]),
                                      bias_init=tf.constant_initializer(dataDict['fc7'][1]))



            layers['drop7'] = tf.nn.dropout(layers['fc7'], keep_prob=keep_prob)


            print layers['drop7'].shape

            featSize=4096
            rnn_inputs = tf.reshape(layers['drop7'], [seqLength, inputDims/seqLength,  featSize])

            layers['rnn_outputs'] = self._LSTM(rnn_inputs, seqLength, weight_decay=0.0, featSize=4096, cellSize=1024)



            layers['drop8'] = tf.nn.dropout(layers['rnn_outputs'], keep_prob=keep_prob)



            if outputDims == dataDict['fc8'][0].shape[1]:
                layers['logits'] = fully_connected_layer(input_tensor=layers['drop8'],
                                          out_dim=outputDims,
                                          name='logits',
                                          weight_decay=weight_decay,
                                          non_linear_fn=None,
                                          weight_init=tf.constant_initializer(dataDict['fc8'][0]),
                                          bias_init=tf.constant_initializer(dataDict['fc8'][1]))
            else:
                layers['logits'] = fully_connected_layer(input_tensor=layers['drop8'],
                                          out_dim=outputDims,
                                          name='logits',
                                          weight_decay=weight_decay,
                                          non_linear_fn=None)


        return layers[return_layer]


    """ Function to return loss calculated on given network """
    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        crossEntropyLoss = tf.losses.sparse_softmax_cross_entropy(labels=labels[:labels.shape[0].value/2],
                        logits=logits[:logits.shape[0].value/2,:])
        return crossEntropyLoss


    def preprocess(self, index, data, labels, size, isTraining):
        return preprocess(index, data,labels, size, isTraining)
