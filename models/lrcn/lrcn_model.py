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


        dataDict = np.load('/z/home/erichof/TF_Activity_Recognition_Framework/models/lrcn/lrcn.npy').tolist()

<<<<<<< HEAD
        with tf.name_scope("my_scope"):
=======
    #    import pdb;pdb.set_trace()
        with tf.name_scope(scope, 'lrcn', [inputs]):#with tf.name_scope("my_scope"):#, values=inputs): #scope, 'lrcn', [inputs]):
>>>>>>> eric-dev

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


<<<<<<< HEAD
=======
            #
            # # Undo the grouping from Caffe
            # # - Half of the filters get passed over half of the input
            # conv4a = tf.nn.conv2d(input=layers['7'][:,:,:,:256],
            #     filter=dataDict['conv4_0'][:,:,:,:256],
            #     strides=[1,1,1,1], padding='SAME',
            #     name='conv4a')
            #
            #
            # conv4b = tf.nn.conv2d(input=layers['7'][:,:,:,256:],
            #     filter=dataDict['conv4_0'][:,:,:,256:],
            #     strides=[1,1,1,1], padding='SAME',
            #     name='conv4a')
            #
            #
            # layers['8'] = tf.nn.relu(
            #                 tf.nn.bias_add(
            #                     tf.concat([conv4a, conv4b], 3, name='conv4'),           #however you concatenate that in tensorflow = [256, 5, 5, 48] + [256, 5, 5, 48] = [512, 5, 5, 48]
            #                     dataDict['conv4_1']),
            #                 'relu4')
            #
            # print('8: ', layers['8'].shape)
            #
            #
            #
            #



            input_slices = array_ops.split(layers['conv3'], 2, axis=3)
            kernel_slices = array_ops.split(dataDict['conv4_0'], 2, axis=3)
            output_slices = [tf.nn.conv2d(
                input=input_slice,
                filter=kernel_slice,
                strides=[1,1,1,1],
                padding='SAME')
                for input_slice, kernel_slice in zip(input_slices, kernel_slices)]
            outputs = array_ops.concat(output_slices, axis=3)
            #print outputs.shapetf.nn.bias_add(outputs,dataDict['conv4_1'])
            layers['conv4_0'] =tf.nn.bias_add(outputs,dataDict['conv4_1'])
            layers['conv4'] = tf.nn.relu(layers['conv4_0'], 'relu4')




#  https://github.com/tensorflow/tensorflow/pull/10482/files#diff-26aa645fdaefe1f89103555b9c0da70eL433


            # # Undo the grouping from Caffe
            # # - Half of the filters get passed over half of the input
            # conv5a = tf.nn.conv2d(input=layers['8'][:,:,:,:256],
            #     filter=dataDict['conv5_0'][:,:,:,:192],
            #     strides=[1,1,1,1], padding='SAME',
            #     name='conv5a')
            #
            #
            # conv5b = tf.nn.conv2d(input=layers['8'][:,:,:,256:],
            #     filter=dataDict['conv5_0'][:,:,:,192:],
            #     strides=[1,1,1,1], padding='SAME',
            #     name='conv5a')
            #
            #
            # layers['9'] = tf.nn.relu(
            #                 tf.nn.bias_add(
            #                     tf.concat([conv5a, conv5b], 3, name='conv5'),           #however you concatenate that in tensorflow = [256, 5, 5, 48] + [256, 5, 5, 48] = [512, 5, 5, 48]
            #                     dataDict['conv5_1']),
            #                 'relu5')
            #
            # print('9: ', layers['9'].shape)
            #
            #
            #
            #




            input_slices = array_ops.split(layers['conv4'], 2, axis=3)
            kernel_slices = array_ops.split(dataDict['conv5_0'], 2, axis=3)
            output_slices = [tf.nn.conv2d(
                input=input_slice,
                filter=kernel_slice,
                strides=[1,1,1,1],
                padding='SAME')
                for input_slice, kernel_slice in zip(input_slices, kernel_slices)]
            outputs = array_ops.concat(output_slices, axis=3)
    #        print outputs.shape
            layers['conv5'] = tf.nn.relu(tf.nn.bias_add(outputs,dataDict['conv5_1']), 'relu5')









            layers['pool5'] = tf.nn.max_pool(value=layers['conv5'], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')


            batch_size = layers['pool5'].get_shape().as_list()[0]
        #    print(batch_size)
            dense_dim = dataDict['fc6_0'].get_shape().as_list()[0]
    #        print(dense_dim)
        #    print("fc6 ", layers['10'].get_shape())
            fc6 = tf.reshape(layers['pool5'], [batch_size, dense_dim])

            layers['fc6'] = tf.nn.relu(tf.matmul(fc6, dataDict['fc6_0']) + dataDict['fc6_1'], 'relu6')
            print isTraining
            with tf.device('/gpu:'+str(cpuId)):
                if isTraining:
                    layers['fc6'] = tf.reshape(layers['fc6'], shape = [1, seqLength, 4096])
                else:
                    layers['fc6'] = tf.reshape(layers['fc6'], shape=[10,seqLength,4096])
>>>>>>> eric-dev


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
        labels = tf.tile(labels, logits.shape[0].value/labels.shape[0].value)
        labels = tf.cast(labels, tf.int64)
        
        crossEntropyLoss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                        logits=logits)

        return crossEntropyLoss
