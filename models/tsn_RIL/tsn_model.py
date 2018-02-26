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


class TSN_RIL():
    def __init__(self, input_dims, output_dims, exp_name, num_seg=3, init=False, verbose=True, model_alpha=1.0, input_alpha=1.0, resample_frames=20):
        self.name            = 'tsn_RIL'
        self.verbose         = verbose
        self.input_dims      = input_dims
        self.output_dims     = output_dims
        self.exp_name        = exp_name
        self.num_seg         = num_seg
        self.init            = init
        self.alpha           = model_alpha
        self.input_alpha     = input_alpha
        self.resample_frames = resample_frames
        self.dropout         = 0.8

        if self.verbose:
            print('TSN_RIL initialized')

    def _extraction_layer(self, inputs, params, model_input, L):
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
        sample_alpha_tick = tf.exp(-tf.nn.relu(params[0]))

        # Extract shape of input signal
        frames, shp_h, shp_w, channel = inputs.get_shape().as_list()

        # Generate indices for output
        output_idx = tf.range(start=1., limit=float(L)+1., delta=1., dtype=tf.float32)

        output_idx = tf.slice(output_idx, [0],[L])

        # Sampling parameter scaling to match inputs temporal dimension
        alpha_tick = sample_alpha_tick * tf.cast(model_input, tf.float32) / (float(L))

        # Include sampling parameter to correct output indices
        output_idx = tf.multiply(tf.tile([alpha_tick], [L]), output_idx)

        # Clip output index values to >= 1 and <=N (valid cases only)
        output_idx = tf.clip_by_value(output_idx, 1., tf.cast(model_input, tf.float32))

        # Create x0 and x1 float
        x0 = tf.clip_by_value(tf.floor(output_idx), 1., tf.cast(model_input, tf.float32)-1.)
        x1 = tf.clip_by_value(tf.floor(output_idx+1.), 2., tf.cast(model_input, tf.float32))


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

    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining):
        return preprocess_tfrecords(input_data_tensor, frames, height, width, channel, size, label, istraining, self.num_seg, self.input_dims)

    def apply_rain(self, inputs, batch_size, input_dims, params, model_input, L):
        input_data_list = []
        for i in range(batch_size):
            start_ind = i*input_dims
            input_seg_list = []
            for j in range(input_dims/model_input):
                indices = tf.range(start_ind+j*model_input, start_ind+(j+1)*model_input)
                #input_seg_data = inputs[start_ind+j*model_input:start_ind+(j+1)*model_input,:,:,:]
                input_seg_data = tf.gather(inputs, indices)
                input_seg_data = self._extraction_layer(inputs=input_seg_data,
                                                        params=params, model_input=model_input, L=L)
                input_seg_list.append(input_seg_data)

            input_data_tensor = tf.stack(input_seg_list, 1)
            #input_data_tensor = tf.reshape(input_data_tensor, [L*(input_dims/model_input), 224, 224, 3])
            input_data_tensor = tf.expand_dims(input_data_tensor, 0)
            input_data_list.append(input_data_tensor)

        out_dim = batch_size*(input_dims/model_input)*L
        input_data_tensor = tf.concat(input_data_list, 0)
        input_data_tensor = tf.reshape(input_data_tensor, [out_dim, 224, 224, 3])

        return input_data_tensor, out_dim

    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, dropout_rate = 0.8, return_layer=['logits'], weight_decay=0.0005):
        if self.verbose:
            print('Generating TSN network')

        input_dim_list = inputs.get_shape().as_list()
        if len(input_dim_list) > 4:
            new_dim_list = [input_dim_list[0]*input_dim_list[1]]
            new_dim_list.extend(input_dim_list[2:])
            inputs = tf.reshape(inputs, new_dim_list)

        print('inference input: ', inputs)

        with tf.name_scope(scope, 'tsn_RIL', [inputs]):
            layers = {}
            layers['Parameterization_Variables'] = [tf.get_variable('alpha',shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.alpha))]
            layers['RAINlayer'], out_dim = self.apply_rain(inputs, input_dim_list[0], input_dim_list[1],
                                                       params=layers['Parameterization_Variables'], model_input=input_dims/3, L=self.resample_frames)

            layers['base'] = BNInception(layers['RAINlayer'],
                                     dropout_rate=dropout_rate,
                                     num_classes=output_dims,
                                     is_training=is_training,
                                     is_init=self.init,
                                     weight_decay=weight_decay,
                                     scope=scope)

            layers['logits'] = tf.reduce_mean(tf.reshape(layers['base'], [out_dim/self.num_seg, self.num_seg, layers['base'].shape[-1].value]), 1)

        if len(input_dim_list) > 4:
                output_dim_list = layers['logits'].get_shape().as_list()
                new_dim_list = [input_dim_list[0], output_dim_list[0]/input_dim_list[0]]
                new_dim_list.extend(output_dim_list[1:])
                layers['logits'] = tf.expand_dims(layers['logits'], 0)
                layers['logits'] = tf.reshape(layers['logits'], new_dim_list)

        return [layers[x] for x in return_layer]
