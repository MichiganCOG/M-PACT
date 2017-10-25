""" THIS FILE SUPPORTS THE GENERATION OF NETWORK SKELETONS """

# Import tensorflow modules
import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn

from resnet_preprocessing import preprocess

# Import math and numpy modules
import numpy as np

# Import HDF5
import h5py

import os

import sys
sys.path.append('../..')
from layers_utils import *

from load_dataset import load_dataset as load_data



class ResNet():

    def __init__(self, verbose=True):
        self.verbose=verbose
        self.name = 'resnet'
        print "resnet initialized"

    def _LSTM(self, inputs, seqLength, featSize, cellSize=1024, isTraining=False):

        # Unstack input tensor to match shape:
        # list of n_time_steps items, each item of size (batch_size x featSize)
        inputs = tf.reshape(inputs, [-1,1,featSize])
        inputs = tf.unstack(inputs, seqLength, axis=0)

        # LSTM cell definition
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(cellSize)
        outputs, states = static_rnn(lstm_cell, inputs, dtype=tf.float32)

        # Condense output shape from:
        # list of n_time_steps itmes, each item of size (batch_size x cellSize)
        # To:
        # Tensor: [(n_time_steps x 1), cellSize] (Specific to our case)
        outputs = tf.stack(outputs)
        outputs = tf.reshape(outputs,[-1,cellSize])


        return outputs

    def _conv_block(self, nFilters, kernelSize, name, layerNumbers, inputLayer, dataDict, strides=2, weight_decay=0.0):

        layers = {}

        # Conv block
        layers[layerNumbers[0]] = conv_layer(input_tensor=inputLayer,
                filter_dims=[1,1,nFilters[0]], stride_dims=[strides,strides],
                padding = 'VALID',
                name='res'+name+'_branch2a',
                kernel_init=tf.constant_initializer(dataDict['res'+name+'_branch2a']['res'+name+'_branch2a_W:0'].value),
                bias_init=tf.constant_initializer(dataDict['res'+name+'_branch2a']['res'+name+'_branch2a_b:0'].value),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layerNumbers[1]] = tf.layers.batch_normalization(layers[layerNumbers[0]],
                name='bn'+name+'_branch2a',
                moving_mean_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2a']['bn'+name+'_branch2a_running_mean:0'].value),
                moving_variance_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2a']['bn'+name+'_branch2a_running_std:0'].value),
                beta_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2a']['bn'+name+'_branch2a_beta:0'].value),
                gamma_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2a']['bn'+name+'_branch2a_gamma:0'].value))

        layers[layerNumbers[2]] = conv_layer(input_tensor=tf.nn.relu(layers[layerNumbers[1]]),
                filter_dims=[kernelSize, kernelSize, nFilters[1]], padding='SAME',
                name='res'+name+'_branch2b',
                kernel_init=tf.constant_initializer(dataDict['res'+name+'_branch2b']['res'+name+'_branch2b_W:0'].value),
                bias_init=tf.constant_initializer(dataDict['res'+name+'_branch2b']['res'+name+'_branch2b_b:0'].value),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layerNumbers[3]] = tf.layers.batch_normalization(layers[layerNumbers[2]],
                name='bn'+name+'_branch2b',
                moving_mean_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2b']['bn'+name+'_branch2b_running_mean:0'].value),
                moving_variance_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2b']['bn'+name+'_branch2b_running_std:0'].value),
                beta_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2b']['bn'+name+'_branch2b_beta:0'].value),
                gamma_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2b']['bn'+name+'_branch2b_gamma:0'].value))

        layers[layerNumbers[4]] = conv_layer(input_tensor=tf.nn.relu(layers[layerNumbers[3]]),
                filter_dims=[1,1,nFilters[2]], padding = 'VALID',
                name='res'+name+'_branch2c',
                kernel_init=tf.constant_initializer(dataDict['res'+name+'_branch2c']['res'+name+'_branch2c_W:0'].value),
                bias_init=tf.constant_initializer(dataDict['res'+name+'_branch2c']['res'+name+'_branch2c_b:0'].value),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layerNumbers[5]] = tf.layers.batch_normalization(layers[layerNumbers[4]],
                name='bn'+name+'_branch2c',
                moving_mean_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2c']['bn'+name+'_branch2c_running_mean:0'].value),
                moving_variance_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2c']['bn'+name+'_branch2c_running_std:0'].value),
                beta_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2c']['bn'+name+'_branch2c_beta:0'].value),
                gamma_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2c']['bn'+name+'_branch2c_gamma:0'].value))

        # Shortcuts
        layers[layerNumbers[6]] = conv_layer(input_tensor=inputLayer,
                filter_dims=[1,1,nFilters[2]], stride_dims=[strides, strides],
                padding = 'VALID',
                name='res'+name+'_branch1',
                kernel_init=tf.constant_initializer(dataDict['res'+name+'_branch1']['res'+name+'_branch1_W:0'].value),
                bias_init=tf.constant_initializer(dataDict['res'+name+'_branch1']['res'+name+'_branch1_b:0'].value),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layerNumbers[7]] = tf.layers.batch_normalization(layers[layerNumbers[6]],
                name='bn'+name+'_branch1',
                moving_mean_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch1']['bn'+name+'_branch1_running_mean:0'].value),
                moving_variance_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch1']['bn'+name+'_branch1_running_std:0'].value),
                beta_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch1']['bn'+name+'_branch1_beta:0'].value),
                gamma_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch1']['bn'+name+'_branch1_gamma:0'].value))

        # END OF CONV BLOCK

        layers[layerNumbers[8]] = tf.nn.relu(tf.add(layers[layerNumbers[5]],layers[layerNumbers[7]]))

        return layers


    def _identity_block(self, nFilters, kernelSize, name, layerNumbers, inputLayer, dataDict, weight_decay=0.0):

        layers = {}

        # Identity block
        layers[layerNumbers[0]] = conv_layer(input_tensor=inputLayer,
                filter_dims=[1, 1, nFilters[0]], padding='VALID',
                name='res'+name+'_branch2a',
                kernel_init=tf.constant_initializer(dataDict['res'+name+'_branch2a']['res'+name+'_branch2a_W:0'].value),
                bias_init=tf.constant_initializer(dataDict['res'+name+'_branch2a']['res'+name+'_branch2a_b:0'].value),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layerNumbers[1]] = tf.layers.batch_normalization(layers[layerNumbers[0]],
                name='bn'+name+'_branch2a',
                moving_mean_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2a']['bn'+name+'_branch2a_running_mean:0'].value),
                moving_variance_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2a']['bn'+name+'_branch2a_running_std:0'].value),
                beta_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2a']['bn'+name+'_branch2a_beta:0'].value),
                gamma_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2a']['bn'+name+'_branch2a_gamma:0'].value))

        layers[layerNumbers[2]] = conv_layer(input_tensor=tf.nn.relu(layers[layerNumbers[1]]),
                filter_dims=[kernelSize, kernelSize, nFilters[1]], padding='SAME',
                name='res'+name+'_branch2b',
                kernel_init=tf.constant_initializer(dataDict['res'+name+'_branch2b']['res'+name+'_branch2b_W:0'].value),
                bias_init=tf.constant_initializer(dataDict['res'+name+'_branch2b']['res'+name+'_branch2b_b:0'].value),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layerNumbers[3]] = tf.layers.batch_normalization(layers[layerNumbers[2]],
                name='bn'+name+'_branch2b',
                moving_mean_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2b']['bn'+name+'_branch2b_running_mean:0'].value),
                moving_variance_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2b']['bn'+name+'_branch2b_running_std:0'].value),
                beta_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2b']['bn'+name+'_branch2b_beta:0'].value),
                gamma_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2b']['bn'+name+'_branch2b_gamma:0'].value))

        layers[layerNumbers[4]] = conv_layer(input_tensor=tf.nn.relu(layers[layerNumbers[3]]),
                filter_dims=[1,1,nFilters[2]], padding='VALID',
                name='res'+name+'_branch2c',
                kernel_init=tf.constant_initializer(dataDict['res'+name+'_branch2c']['res'+name+'_branch2c_W:0'].value),
                bias_init=tf.constant_initializer(dataDict['res'+name+'_branch2c']['res'+name+'_branch2c_b:0'].value),
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layerNumbers[5]] = tf.layers.batch_normalization(layers[layerNumbers[4]],
                name='bn'+name+'_branch2c',
                moving_mean_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2c']['bn'+name+'_branch2c_running_mean:0'].value),
                moving_variance_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2c']['bn'+name+'_branch2c_running_std:0'].value),
                beta_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2c']['bn'+name+'_branch2c_beta:0'].value),
                gamma_initializer=tf.constant_initializer(dataDict['bn'+name+'_branch2c']['bn'+name+'_branch2c_gamma:0'].value))

        # END OF IDENTITY BLOCK

        layers[layerNumbers[6]] = tf.nn.relu(tf.add(layers[layerNumbers[5]],inputLayer))

        return layers

    def inference(self, x, labels, isTraining, inputDims, outputDims, seqLength, scope, dropoutRate = 0.5, return_layer='logits', dataDict=None, cpuId = 0, weight_decay=0.0):#x, isTraining, dataDict, seqLength, outputDims):#gen_resnet50_baseline1_network(x, isTraining, dataDict, seqLength, outputDims):

        ############################################################################
        #                       Creating ResNet50 Network Layers                   #
        ############################################################################

        print('Generating RESNET network layers')
        dataDict = h5py.File('models/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5','r')
        layers = {}

        layers['1'] = conv_layer(input_tensor=x,
                filter_dims=[7, 7, 64], stride_dims=[2,2],
                padding = 'VALID',
                name='conv1',
                kernel_init=tf.constant_initializer(dataDict['conv1']['conv1_W:0'].value),
                bias_init=tf.constant_initializer(dataDict['conv1']['conv1_b:0'].value),
                weight_decay = weight_decay, non_linear_fn=None)

        layers['2'] = tf.layers.batch_normalization(layers['1'],
                name='bn_conv1',
                moving_mean_initializer=tf.constant_initializer(dataDict['bn_conv1']['bn_conv1_running_mean:0'].value),
                moving_variance_initializer=tf.constant_initializer(dataDict['bn_conv1']['bn_conv1_running_std:0'].value),
                beta_initializer=tf.constant_initializer(dataDict['bn_conv1']['bn_conv1_beta:0'].value),
                gamma_initializer=tf.constant_initializer(dataDict['bn_conv1']['bn_conv1_gamma:0'].value))

        layers['3'] = max_pool_layer(tf.nn.relu(layers['2']),
                filter_dims=[3, 3], stride_dims=[2,2], name='pool1', padding='VALID')


        layers.update(self._conv_block([64,64,256], kernelSize=3, name='2a', layerNumbers=['4','5','6','7','8','9','10','11','12'],
                        inputLayer=layers['3'], strides=1, dataDict=dataDict))

        layers.update(self._identity_block([64,64,256], kernelSize=3, name='2b', layerNumbers=['13','14','15','16','17','18','19'],
                        inputLayer=layers['12'], dataDict=dataDict))

        layers.update(self._identity_block([64,64,256], kernelSize=3, name='2c', layerNumbers=['20','21','22','23','24','25','26'],
                        inputLayer=layers['19'], dataDict=dataDict))

        #########
        layers.update(self._conv_block([128,128,512], kernelSize=3, name='3a', layerNumbers=['27','28','29','30','31','32','33','34','35'],
                        inputLayer=layers['26'], dataDict=dataDict))

        layers.update(self._identity_block([128,128,512], kernelSize=3, name='3b', layerNumbers=['36','37','38','39','40','41','42'],
                        inputLayer=layers['35'], dataDict=dataDict))

        layers.update(self._identity_block([128,128,512], kernelSize=3, name='3c', layerNumbers=['43','44','45','46','47','48','49'],
                        inputLayer=layers['42'], dataDict=dataDict))

        layers.update(self._identity_block([128,128,512], kernelSize=3, name='3d', layerNumbers=['50','51','52','53','54','55','56'],
                        inputLayer=layers['49'], dataDict=dataDict))

        #########
        layers.update(self._conv_block([256,256,1024], kernelSize=3, name='4a', layerNumbers=['57','58','59','60','61','62','63','64','65'],
                        inputLayer=layers['56'], dataDict=dataDict))

        layers.update(self._identity_block([256,256,1024], kernelSize=3, name='4b', layerNumbers=['66','67','68','69','70','71','72'],
                        inputLayer=layers['65'], dataDict=dataDict))

        layers.update(self._identity_block([256,256,1024], kernelSize=3, name='4c', layerNumbers=['73','74','75','76','77','78','79'],
                        inputLayer=layers['72'], dataDict=dataDict))

        layers.update(self._identity_block([256,256,1024], kernelSize=3, name='4d', layerNumbers=['80','81','82','83','84','85','86'],
                        inputLayer=layers['79'], dataDict=dataDict))

        layers.update(self._identity_block([256,256,1024], kernelSize=3, name='4e', layerNumbers=['87','88','89','90','91','92','93'],
                        inputLayer=layers['86'], dataDict=dataDict))

        layers.update(self._identity_block([256,256,1024], kernelSize=3, name='4f', layerNumbers=['94','95','96','97','98','99','100'],
                        inputLayer=layers['93'], dataDict=dataDict))

        #########
        layers.update(self._conv_block([512,512,2048], kernelSize=3, name='5a', layerNumbers=['101','102','103','104','105','106','107','108','109'],
                        inputLayer=layers['100'], dataDict=dataDict))

        layers.update(self._identity_block([512,512,2048], kernelSize=3, name='5b', layerNumbers=['110','111','112','113','114','115','116'],
                        inputLayer=layers['109'], dataDict=dataDict))

        layers.update(self._identity_block([512,512,2048], kernelSize=3, name='5c', layerNumbers=['117','118','119','120','121','122','123'],
                        inputLayer=layers['116'], dataDict=dataDict))

        layers['124'] = tf.reduce_mean(layers['123'], reduction_indices=[1,2], name='avg_pool')
        layers['125'] = self._LSTM(layers['124'], seqLength, featSize=2048, cellSize=512, isTraining=isTraining)

        layers['126'] = tf.layers.dropout(layers['125'], training=isTraining, rate=0.5)

        layers['logits'] = fully_connected_layer(input_tensor=layers['126'],
                out_dim=outputDims, non_linear_fn=None,
                name='logits', weight_decay=weight_decay)

        return layers[return_layer]



    def preprocess(self, index, data, labels, size, isTraining):
        return preprocess(index, data,labels, size, isTraining)

    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        crossEntropyLoss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                        logits=logits)
        return crossEntropyLoss













if __name__=="__main__":

    import os
    x = tf.placeholder(tf.float32, shape=(None, 224,224,3))
    y = tf.placeholder(tf.int32, [None])
    path = os.path.join('/z/home/madantrg/RILCode/Code_TF_ND/ExperimentBaseline','resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    data_dict = h5py.File(path,'r')

    network = _gen_resnet50_baseline1_network(x, True, data_dict, 35, 51)
    import pdb; pdb.set_trace()
