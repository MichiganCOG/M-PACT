import h5py
import os
import time
import sys
sys.path.append('../..')

import tensorflow as tf
import numpy      as np

from utils.layers_utils                       import *
from tsn_preprocessing_TFRecords              import preprocess as preprocess_tfrecords
from BNInception                              import BNInception


class TSN_SR():
    def __init__(self, input_dims, output_dims, exp_name, num_seg=3, init=False, model_alpha=1.0, input_alpha=1.0, verbose=True):
        self.name        = 'tsn_sr'
        self.verbose     = verbose
        self.input_dims  = input_dims
        self.output_dims = output_dims
        self.exp_name    = exp_name
        self.num_seg     = num_seg
        self.init        = init
        self.model_alpha = model_alpha
        self.input_alpha = input_alpha
        self.dropout     = 0.8
	self.store_alpha = True

        if self.verbose:
            print('TSN_SR initialized')

    def _extraction_layer(self, inputs, params, sets, K, L):
        # Parameter definitions are taken as mean ($\psi(\cdot)$) of input estimates
        sample_alpha_tick = tf.exp(-tf.nn.relu(params[0]))

        # Extract shape of input signal
        frames, shp_h, shp_w, channel = inputs.get_shape().as_list()

        # Generate indices for output
        output_idx = tf.range(start=1., limit=float(L)+1., delta=1., dtype=tf.float32)

        output_idx = tf.slice(output_idx, [0],[L])

        # Sampling parameter scaling to match inputs temporal dimension
        alpha_tick = sample_alpha_tick * tf.cast(K * sets, tf.float32) / (float(L))

        # Include sampling parameter to correct output indices
        output_idx = tf.multiply(tf.tile([alpha_tick], [L]), output_idx)

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

    def loss(self, logits, labels, loss_type):
        labels_dim_list = labels.get_shape().as_list()
        if len(labels_dim_list) > 1:
            labels = tf.reshape(labels, [labels_dim_list[0]*labels_dim_list[1]])

        labels = tf.cast(labels, tf.int32)

        input_dim_list = logits.get_shape().as_list()
        if len(input_dim_list) > 2:
            new_dim_list = [input_dim_list[0]*input_dim_list[1]]
            new_dim_list.extend(input_dim_list[2:])
            logits = tf.reshape(logits, new_dim_list)

        print('loss labels: %r' % labels)
        print('loss logits: %r' % logits)
        #logits = tf.reshape(logits, [self.num_seg, -1, logits.shape[-1].value])
        #logits = tf.reduce_mean(logits, 0)
        #tf.losses.sparse_softmax_cross_entropy(labels=labels[:logits.shape[0].value], logits=logits)
        total_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels[:logits.shape[0].value], logits=logits)

        #softmax_loss = tf.get_collection(tf.GraphKeys.LOSSES)
        #regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #loss = tf.add_n(softmax_loss + regularization_loss, name='loss')
        #total_loss = loss
        return total_loss

    def load_default_weights(self):
        weights = None
        weight_file_name = ''
        if 'init' in self.exp_name:
            self.init = True
            weight_file_name = 'models/tsn/bn_inception_rgb_init.npy'
        elif self.output_dims == 51:
            weight_file_name = 'models/tsn/tsn_pretrained_HMDB51_reordered.npy'
        elif self.output_dims == 101:
            weight_file_name = 'models/tsn/tsn_pretrained_UCF101_reordered.npy'
        elif self.verbose:
            print('Cannot find the weight file')

        if os.path.isfile(weight_file_name):
            weights = np.load(weight_file_name)

        return weights

    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step):
        output, alpha_tensor = preprocess_tfrecords(input_data_tensor, frames, height, width, channel, size, label, istraining, self.num_seg, self.input_dims, self.model_alpha, self.input_alpha, video_step)
        return output, alpha_tensor

    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, dropout_rate = 0.8, return_layer=['logits'], weight_decay=0.0005):
        if self.verbose:
            print('Generating TSN network')

        input_dim_list = inputs.get_shape().as_list()
        if len(input_dim_list) > 4:
            new_dim_list = [input_dim_list[0]*input_dim_list[1]]
            new_dim_list.extend(input_dim_list[2:])
            inputs = tf.reshape(inputs, new_dim_list)

        print('inference input: ', inputs)

        with tf.name_scope(scope, 'tsn_sr', [inputs]):
            layers = {}
            layers['Parameterization_Variables'] = self.store_alpha
            #layers['RIlayer'] = self._extraction_layer(inputs=inputs, params=layers['Parameterization_Variables'], sets=input_dims, L=30, K=1)
            layers['base'] = BNInception(inputs,
                                     dropout_rate=dropout_rate,
                                     num_classes=output_dims,
                                     is_training=is_training,
                                     is_init=self.init,
                                     weight_decay=weight_decay,
                                     scope=scope)
            #layers['logits'] = tf.reduce_mean(tf.reshape(layers['base'], [self.num_seg, input_dims/self.num_seg, layers['base'].shape[-1].value]), 0)
            layers['logits'] = tf.reduce_mean(tf.reshape(layers['base'], [new_dim_list[0]/self.num_seg, self.num_seg, layers['base'].shape[-1].value]), 1)

        if len(input_dim_list) > 4:
                output_dim_list = layers['logits'].get_shape().as_list()
                new_dim_list = [input_dim_list[0], output_dim_list[0]/input_dim_list[0]]
                new_dim_list.extend(output_dim_list[1:])
                layers['logits'] = tf.expand_dims(layers['logits'], 0)
                layers['logits'] = tf.reshape(layers['logits'], new_dim_list)

        return [layers[x] for x in return_layer]

    def _inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, dropout_rate = 0.5, return_layer=['logits'], weight_decay=0.0):
        if self.verbose:
            print('Generating TSN network')

        input_dim_list = inputs.get_shape().as_list()
        if len(input_dim_list) > 4:
            new_dim_list = [input_dim_list[0]*input_dim_list[1]]
            new_dim_list.extend(input_dim_list[2:])
            inputs = tf.reshape(inputs, new_dim_list)

        print('inference input: ', inputs)

        with tf.name_scope(scope, 'tsn', [inputs]):
            layers = {}
            #layers['Parameterization_Variables'] = [tf.get_variable('alpha',shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0.25))]
            #layers['RIlayer'] = self._extraction_layer(inputs=inputs, params=layers['Parameterization_Variables'], sets=input_dims, L=150, K=1)
            layers['conv1/7x7_s2'] = conv_layer(input_tensor=inputs,filter_dims=[7,7,64],stride_dims=[2,2],padding='SAME',name='conv1/7x7_s2',
                                                weight_decay=weight_decay, non_linear_fn=None)
            layers['conv1/7x7_s2_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers['conv1/7x7_s2'],training=is_training,trainable=is_training,
                                                                                 epsilon=1e-5,momentum=0.999,name='conv1/7x7_s2_bn'))
            layers['pool1/3x3_s2'] = max_pool_layer(layers['conv1/7x7_s2_bn'],filter_dims=[3,3],stride_dims=[2,2],name='pool1/3x3_s2',
                                                    padding='VALID')
            layers['conv2/3x3_reduce'] = conv_layer(input_tensor=layers['pool1/3x3_s2'],filter_dims=[1,1,64],stride_dims=[1,1],padding='SAME',
                                                    name='conv2/3x3_reduce',weight_decay=weight_decay,non_linear_fn=None)
            layers['conv2/3x3_reduce_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers['conv2/3x3_reduce'],training=is_training,trainable=False,
                                                                                     epsilon=1e-5,momentum=0.999,name='conv2/3x3_reduce_bn'))
            layers['conv2/3x3'] = conv_layer(input_tensor=layers['conv2/3x3_reduce_bn'],filter_dims=[3,3,192],stride_dims=[1,1],padding='SAME',
                                             name='conv2/3x3',weight_decay=weight_decay,non_linear_fn=None)
            layers['conv2/3x3_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers['conv2/3x3'],training=is_training,trainable=False,
                                                                              epsilon=1e-5,momentum=0.999,name='conv2/3x3_bn'))
            layers['pool2/3x3_s2'] = max_pool_layer(layers['conv2/3x3_bn'],filter_dims=[3,3],stride_dims=[2,2],name='pool2/3x3_s2', padding='VALID')
            layers.update(self._inception_block_1(layers['pool2/3x3_s2'],[64, 64, 64, 64, 96, 96, 32],weight_decay,is_training,'inception_3a'))
            layers.update(self._inception_block_1(layers['inception_3a/output'],[64, 64, 96, 64, 96, 96, 64],weight_decay,is_training,'inception_3b'))
            layers.update(self._inception_block_2(layers['inception_3b/output'],[128, 160, 64, 96, 96],weight_decay,is_training,'inception_3c'))
            layers.update(self._inception_block_1(layers['inception_3c/output'],[224, 64, 96, 96, 128, 128, 128],weight_decay,is_training,'inception_4a'))
            layers.update(self._inception_block_1(layers['inception_4a/output'],[192, 96, 128, 96, 128, 128, 128],weight_decay,is_training,'inception_4b'))
            layers.update(self._inception_block_1(layers['inception_4b/output'],[160, 128, 160, 128, 160, 160, 128],weight_decay,is_training,'inception_4c'))
            layers.update(self._inception_block_1(layers['inception_4c/output'],[96, 128, 192, 160, 192, 192, 128],weight_decay,is_training,'inception_4d'))
            layers.update(self._inception_block_2(layers['inception_4d/output'],[128, 192, 192, 256, 256],weight_decay,is_training,'inception_4e'))
            layers.update(self._inception_block_1(layers['inception_4e/output'],[352, 192, 320, 160, 224, 224, 128],weight_decay,is_training,'inception_5a'))
            layers.update(self._inception_block_1(layers['inception_5a/output'],[352, 192, 320, 192, 224, 224, 128],weight_decay,is_training,'inception_5b'),
                                                  pool='max')

            layers['global_pool'] = avg_pool_layer(layers['inception_5b/output'],filter_dims=[7,7],stride_dims=[1,1],name='global_pool',padding='VALID')
            layers['dropout'] = tf.layers.dropout(layers['global_pool'], training=is_training,rate=self.dropout,name='dropout')
            layers['fc-action'] = fully_connected_layer(input_tensor=layers['dropout'],out_dim=output_dims,non_linear_fn=None,name='fc-action',
                                                        weight_decay=weight_decay)


            layers['logits'] = tf.reduce_mean(tf.reshape(layers['fc-action'], [self.num_seg, new_dim_list[0]/self.num_seg, layers['fc-action'].shape[-1].value]), 0)

            if len(input_dim_list) > 4:
                output_dim_list = layers['logits'].get_shape().as_list()
                new_dim_list = [input_dim_list[0], output_dim_list[0]/input_dim_list[0]]
                new_dim_list.extend(output_dim_list[1:])
                layers['logits'] = tf.expand_dims(layers['logits'], 0)
                layers['logits'] = tf.reshape(layers['logits'], new_dim_list)

        return [layers[x] for x in return_layer]
