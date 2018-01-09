" RESNET-50 + RAIN (INTERP + MEDIAN) v31_3 + LSTM MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "

import os
import sys
import h5py
sys.path.append('../..')

import tensorflow as tf
import numpy      as np

from tensorflow.contrib.rnn          import static_rnn
from layers_utils                    import *
from resnet_preprocessing_TFRecords  import preprocess   as preprocess_tfrecords

class ResNet_RIL_Interp_Median_v31_3():

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
        self.verbose    = verbose
        self.input_dims = input_dims
        self.j          = input_dims / k
        self.name       = 'resnet_RIL_interp_median_model_v31_3'
    
        print "resnet RIL interp median v31_3 initialized"

    def _extraction_layer(self, inputs, params, sets, K, L):
        """
        Args:
            :inputs: Original inputs to the model
            :params: Offset and sampling parameter estimates from Parameterization Network
            :sets:   Number of non overlapping sets obtained from applying a temporal slid
                     ing window over the input
            :K:      Size of temporal sliding window
            :L:      Expected number of output frames

        Return:
            :output: Extracted features
        """

        # Parameter definitions are taken as mean ($\psi(\cdot)$) of input estimates
        sample_phi_tick   = tf.cond(params[0] > 0, lambda: tf.tanh(tf.nn.relu(params[0]))*0.5 + 0.5, lambda: tf.tanh(tf.nn.relu(-params[0]))*0.5 + 0.5)

        # Extract shape of input signal
        frames, shp_h, shp_w, channel = inputs.get_shape().as_list()


        # Offset scaling to match inputs temporal dimension
        sample_phi_tick = sample_phi_tick * tf.cast((sets*K) - L, tf.float32)

        phi_tick = tf.tile([sample_phi_tick], [L])

        # Generate indices for output
        output_idx = tf.range(start=1., limit=float(L)+1., delta=1., dtype=tf.float32)

        output_idx = tf.slice(output_idx, [0],[L])

        # Add offset to the output indices
        output_idx = output_idx + phi_tick

        # Clip output index values to >= 1 and <=N (valid cases only)
        output_idx = tf.clip_by_value(output_idx, 1., tf.cast(sets*K, tf.float32))

        # Create x0 and x1 float
        x0 = tf.clip_by_value(tf.floor(output_idx), 1., tf.cast(sets*K, tf.float32)-1.)
        x1 = tf.clip_by_value(tf.floor(output_idx+1.), 2., tf.cast(sets*K, tf.float32))


        # Deltas :
        d1 = (output_idx - x0)
        d2 = (x1 - x0)
        d1 = tf.reshape(tf.tile(d1, [224*224*3]), [L,224,224,3])
        d2 = tf.reshape(tf.tile(d2, [224*224*3]), [L,224,224,3])

        # Create x0 and x1 indices
        output_idx_0 = tf.cast(tf.floor(output_idx), 'int32')
        output_idx_1 = tf.cast(tf.ceil(output_idx), 'int32')
        output_idx   = tf.cast(output_idx, 'int32')

	# Create y0 and y1 outputs
        output_0 = tf.gather(inputs, output_idx_0-1)
        output_1 = tf.gather(inputs, output_idx_1-1)
        output   = tf.gather(inputs, output_idx-1)

        d3     = output_1 - output_0

        output = tf.add_n([(d1/d2)*d3, output_0])

        output = tf.reshape(output, (L, shp_h, shp_w, channel), name='RIlayeroutput')

        return output




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
        # list of n_time_steps itmes, each item of size (batch_size x cellSize)
        # To:
        # Tensor: [(n_time_steps x 1), cellSize] (Specific to our case)
        lstm_outputs = tf.stack(lstm_outputs)
        lstm_outputs = tf.reshape(lstm_outputs,[-1,cell_size])


        return lstm_outputs

    def _conv_block(self, n_filters, kernel_size, name, layer_numbers, input_layer, strides=2, weight_decay=0.0):
        """
        Args:
            :n_filters:     List detailing the number of filters
            :kernel_size:   List detailing the height and width of the kernel
            :name:          Name of the conv_block branch
            :layer_numbers: List detailing the connecting layer indices
            :input_layer:   Input layer to the conv_block
            :strides:       Integer value for stride between filters
            :weight_decay:  Double value of weight decay

        Return:
            :layers:        Stack of layers
        """

        layers = {}

        # Conv block
        layers[layer_numbers[0]] = conv_layer(input_tensor=input_layer,
                filter_dims=[1,1,n_filters[0]], stride_dims=[strides,strides],
                padding = 'VALID',
                name='res'+name+'_branch2a',
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[1]] = tf.layers.batch_normalization(layers[layer_numbers[0]],
                name='bn'+name+'_branch2a')

        layers[layer_numbers[2]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[1]]),
                filter_dims=[kernel_size, kernel_size, n_filters[1]], padding='SAME',
                name='res'+name+'_branch2b',
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[3]] = tf.layers.batch_normalization(layers[layer_numbers[2]],
                name='bn'+name+'_branch2b')

        layers[layer_numbers[4]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[3]]),
                filter_dims=[1,1,n_filters[2]], padding = 'VALID',
                name='res'+name+'_branch2c',
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[5]] = tf.layers.batch_normalization(layers[layer_numbers[4]],
                name='bn'+name+'_branch2c')

        # Shortcuts
        layers[layer_numbers[6]] = conv_layer(input_tensor=input_layer,
                filter_dims=[1,1,n_filters[2]], stride_dims=[strides, strides],
                padding = 'VALID',
                name='res'+name+'_branch1',
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[7]] = tf.layers.batch_normalization(layers[layer_numbers[6]],
                name='bn'+name+'_branch1')

        # END OF CONV BLOCK

        layers[layer_numbers[8]] = tf.nn.relu(tf.add(layers[layer_numbers[5]],layers[layer_numbers[7]]))

        return layers


    def _identity_block(self, n_filters, kernel_size, name, layer_numbers, input_layer, weight_decay=0.0):
        """
        Args:
            :n_filters:     List detailing the number of filters
            :kernel_size:   List detailing the height and width of the kernel
            :name:          Name of the identity_block branch
            :layer_numbers: List detailing the connecting layer indices
            :input_layer:   Input layer to the identity_block
            :strides:       Integer value for stride between filters
            :weight_decay:  Double value of weight decay

        Return:
            :layers:        Stack of layers
        """

        layers = {}

        # Identity block
        layers[layer_numbers[0]] = conv_layer(input_tensor=input_layer,
                filter_dims=[1, 1, n_filters[0]], padding='VALID',
                name='res'+name+'_branch2a',
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[1]] = tf.layers.batch_normalization(layers[layer_numbers[0]],
                name='bn'+name+'_branch2a')

        layers[layer_numbers[2]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[1]]),
                filter_dims=[kernel_size, kernel_size, n_filters[1]], padding='SAME',
                name='res'+name+'_branch2b',
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[3]] = tf.layers.batch_normalization(layers[layer_numbers[2]],
                name='bn'+name+'_branch2b')

        layers[layer_numbers[4]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[3]]),
                filter_dims=[1,1,n_filters[2]], padding='VALID',
                name='res'+name+'_branch2c',
                weight_decay = weight_decay, non_linear_fn=None)

        layers[layer_numbers[5]] = tf.layers.batch_normalization(layers[layer_numbers[4]],
                name='bn'+name+'_branch2c')

        # END OF IDENTITY BLOCK

        layers[layer_numbers[6]] = tf.nn.relu(tf.add(layers[layer_numbers[5]],input_layer))

        return layers

    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, dropout_rate = 0.5, return_layer=['logits'], weight_decay=0.0):
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
            :weight_decay: Double value of weight decay

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """

        ############################################################################
        #               Creating ResNet50 + RAIN (interp) + LSTM Network Layers                   #
        ############################################################################

        if self.verbose:
            print('Generating RESNET RAIN INTERP MEDIAN v31_3 network layers')

        # END IF


        with tf.name_scope(scope, 'resnet', [inputs]):
            layers = {}

            # Input shape:  [(K frames in a set x J number of sets) x Height x Width x Channels]
            # Output shape: [(K frames in a set x J number of sets) x Height x Width x 32]

            ############################################################################
            #                           Parameterization Network                       #
            ############################################################################

            layers['Parameterization_Variables'] = [tf.get_variable('phi',shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0.5))]

            layers['RIlayer'] = self._extraction_layer(inputs=inputs,
                                                       params=layers['Parameterization_Variables'],
                                                       sets=self.j, L=seq_length, K=self.k)

            ############################################################################

            layers['1'] = conv_layer(input_tensor=layers['RIlayer'],
                    filter_dims=[7, 7, 64], stride_dims=[2,2],
                    padding = 'VALID',
                    name='conv1',
                    weight_decay = weight_decay, non_linear_fn=None)

            layers['2'] = tf.layers.batch_normalization(layers['1'],
                    name='bn_conv1')

            layers['3'] = max_pool_layer(tf.nn.relu(layers['2']),
                                         filter_dims=[3, 3], stride_dims=[2,2],
                                         name='pool1', padding='VALID')

            layers.update(self._conv_block([64,64,256], kernel_size=3, name='2a', layer_numbers=['4','5','6','7','8','9','10','11','12'],
                            input_layer=layers['3'], strides=1))

            layers.update(self._identity_block([64,64,256], kernel_size=3, name='2b', layer_numbers=['13','14','15','16','17','18','19'],
                            input_layer=layers['12']))

            layers.update(self._identity_block([64,64,256], kernel_size=3, name='2c', layer_numbers=['20','21','22','23','24','25','26'],
                            input_layer=layers['19']))

            #########
            layers.update(self._conv_block([128,128,512], kernel_size=3, name='3a', layer_numbers=['27','28','29','30','31','32','33','34','35'],
                            input_layer=layers['26']))

            layers.update(self._identity_block([128,128,512], kernel_size=3, name='3b', layer_numbers=['36','37','38','39','40','41','42'],
                            input_layer=layers['35']))

            layers.update(self._identity_block([128,128,512], kernel_size=3, name='3c', layer_numbers=['43','44','45','46','47','48','49'],
                            input_layer=layers['42']))

            layers.update(self._identity_block([128,128,512], kernel_size=3, name='3d', layer_numbers=['50','51','52','53','54','55','56'],
                            input_layer=layers['49']))

            #########
            layers.update(self._conv_block([256,256,1024], kernel_size=3, name='4a', layer_numbers=['57','58','59','60','61','62','63','64','65'],
                            input_layer=layers['56']))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4b', layer_numbers=['66','67','68','69','70','71','72'],
                            input_layer=layers['65']))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4c', layer_numbers=['73','74','75','76','77','78','79'],
                            input_layer=layers['72']))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4d', layer_numbers=['80','81','82','83','84','85','86'],
                            input_layer=layers['79']))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4e', layer_numbers=['87','88','89','90','91','92','93'],
                            input_layer=layers['86']))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4f', layer_numbers=['94','95','96','97','98','99','100'],
                            input_layer=layers['93']))

            #########
            layers.update(self._conv_block([512,512,2048], kernel_size=3, name='5a', layer_numbers=['101','102','103','104','105','106','107','108','109'],
                            input_layer=layers['100']))

            layers.update(self._identity_block([512,512,2048], kernel_size=3, name='5b', layer_numbers=['110','111','112','113','114','115','116'],
                            input_layer=layers['109']))

            layers.update(self._identity_block([512,512,2048], kernel_size=3, name='5c', layer_numbers=['117','118','119','120','121','122','123'],
                            input_layer=layers['116']))

            layers['124'] = tf.reduce_mean(layers['123'], reduction_indices=[1,2], name='avg_pool')

            layers['125'] = self._LSTM(layers['124'], seq_length, feat_size=2048, cell_size=512)

            layers['126'] = tf.layers.dropout(layers['125'], training=is_training, rate=0.5)

            layers['logits'] = fully_connected_layer(input_tensor=layers['126'],
                                                     out_dim=output_dims, non_linear_fn=None,
                                                     name='logits', weight_decay=weight_decay)

            # END WITH

        return [layers[x] for x in return_layer]



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
        
        Return:
            Pointer to preprocessing function of current model
        """
        return preprocess_tfrecords(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining)

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

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                    logits=logits)
        return cross_entropy_loss
