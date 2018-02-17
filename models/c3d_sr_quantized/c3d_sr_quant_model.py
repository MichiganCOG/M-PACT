" C3D MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "


"""
Model weights found at https://github.com/hx173149/C3D-tensorflow. The model used was C3D UCF101 TF train - finetuning on UCF101 split1 use C3D sports1M model by @ hdx173149.
"""

import sys
#sys.path.append('utils')

import tensorflow as tf
import numpy      as np

from utils.layers_utils                import *
from c3d_preprocessing_TFRecords import preprocess   as preprocess_tfrecords


class C3D_SR_QUANT():
    def __init__(self, input_dims, clip_length, num_vids, num_epochs, batch_size, num_clips, model_alpha=1.0, input_alpha=1.0, num_gpus=1, verbose=True):
        """
        Args:
            :verbose: Setting verbose command
        """
        self.verbose=verbose
        self.clip_length = clip_length
        self.model_alpha=model_alpha
        self.num_vids = num_vids
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_clips = num_clips
        self.input_alpha = input_alpha
        self.num_gpus = num_gpus
        self.input_dims = input_dims
        self.name = 'c3d_sr_quant'

        if verbose:
            print "C3D Model Initialized"


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
        #                       Creating C3D Network Layers                        #
        ############################################################################

        if self.verbose:
            print('Generating C3D network layers')

        # END IF

        with tf.name_scope(scope, 'c3d', [inputs]):
            layers = {}

            alpha_list = tf.convert_to_tensor([0.4, 0.8, 1.5, 2.5])
            tracker = [v for v in tf.global_variables() if v.name == 'my_scope/global_step:0'][0]
            curr_epoch = tracker * self.num_gpus * self.batch_size / (self.num_vids * self.num_clips)

            alpha_ind = tf.mod(curr_epoch, 4)

            layers['Parameterization_Variables'] = alpha_list[alpha_ind] * tf.cast(self.clip_length, tf.float32) / float(self.input_dims)

            layers['conv1'] = conv3d_layer(input_tensor=inputs,
                    filter_dims=[3, 3, 3, 64],
                    name='c1',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool1'] = max_pool3d_layer(layers['conv1'],
                                                 filter_dims=[1,2,2], stride_dims=[1,2,2],
                                                 name='pool1')

            layers['conv2'] = conv3d_layer(input_tensor=layers['pool1'],
                    filter_dims=[3, 3, 3, 128],
                    name='c2',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool2'] = max_pool3d_layer(layers['conv2'],
                                                 filter_dims=[2,2,2], stride_dims=[2,2,2],
                                                 name='pool2')

            layers['conv3a'] = conv3d_layer(input_tensor=layers['pool2'],
                    filter_dims=[3, 3, 3, 256],
                    name='c3a',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['conv3b'] = conv3d_layer(input_tensor=layers['conv3a'],
                    filter_dims=[3, 3, 3, 256],
                    name='c3b',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool3'] = max_pool3d_layer(layers['conv3b'],
                                                 filter_dims=[2,2,2], stride_dims=[2,2,2],
                                                 name='pool3')

            layers['conv4a'] = conv3d_layer(input_tensor=layers['pool3'],
                    filter_dims=[3, 3, 3, 512],
                    name='c4a',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['conv4b'] = conv3d_layer(input_tensor=layers['conv4a'],
                    filter_dims=[3, 3, 3, 512],
                    name='c4b',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool4'] = max_pool3d_layer(layers['conv4b'],
                                                 filter_dims=[2,2,2], stride_dims=[2,2,2],
                                                 name='pool4')

            layers['conv5a'] = conv3d_layer(input_tensor=layers['pool4'],
                    filter_dims=[3, 3, 3, 512],
                    name='c5a',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['conv5b'] = conv3d_layer(input_tensor=layers['conv5a'],
                    filter_dims=[3, 3, 3, 512],
                    name='c5b',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool5'] = max_pool3d_layer(layers['conv5b'],
                                                 filter_dims=[2,2,2], stride_dims=[2,2,2],
                                                 name='pool5')


            # Uncomment to use sports1m_finetuned_ucf101.model (aka c3d.npy)
            #layers['transpose'] = tf.transpose(layers['pool5'], perm=[0,1,4,2,3], name='transpose')
            #layers['reshape'] = tf.reshape(layers['transpose'], shape=[tf.shape(inputs)[0], 8192], name='reshape')

            # Uncomment to use c3d_Sports1M.npy
            layers['reshape'] = tf.reshape(layers['pool5'], shape=[tf.shape(inputs)[0], 8192], name='reshape')

            layers['dense1'] = fully_connected_layer(input_tensor=layers['reshape'],
                                                     out_dim=4096, non_linear_fn=tf.nn.relu,
                                                     name='d1', weight_decay=weight_decay)

            layers['dropout1'] = tf.layers.dropout(layers['dense1'], training=is_training, rate=dropout_rate)

            layers['dense2'] = fully_connected_layer(input_tensor=layers['dropout1'],
                                                     out_dim=4096, non_linear_fn=tf.nn.relu,
                                                     name='d2', weight_decay=weight_decay)

            layers['dropout2'] = tf.layers.dropout(layers['dense2'], training=is_training, rate=dropout_rate)

            layers['logits'] = tf.expand_dims(fully_connected_layer(input_tensor=layers['dropout2'],
                                                     out_dim=output_dims, non_linear_fn=None,
                                                     name='out', weight_decay=weight_decay), 1)

        return [layers[x] for x in return_layer]

    def load_default_weights(self):
        """
        return: Numpy dictionary containing the names and values of the weight tensors used to initialize this model
        """
        return np.load('models/c3d/c3d_Sports1M.npy')
        # REMOVE pool5 TRANSPOSE FOR SPORTS1M!!!

    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining):
        """
        Args:
            :index:       Integer indicating the index of video frame from the text file containing video lists
            :data:        Data loaded from HDF5 files
            :labels:      Labels for loaded data
            :size:        List detailing values of height and width for final frames
            :is_training: Boolean value indication phase (TRAIN OR TEST)
        """
        output, alpha_tensor = preprocess_tfrecords(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, self.model_alpha, self.input_alpha, self.num_vids, self.num_epochs, self.batch_size, self.num_clips, self.num_gpus)
        return output


    """ Function to return loss calculated on given network """
    def loss(self, logits, labels, loss_type):
        """
        Args:
            :logits: Unscaled logits returned from final layer in model
            :labels: True labels corresponding to loaded data
        """
        labels = tf.cast(labels, tf.int64)

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                    logits=logits)
        return cross_entropy_loss
