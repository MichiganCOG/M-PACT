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


class C3D_ALPHA_EXP():
    def __init__(self, model_alpha=0.0, input_alpha=1.0, resample_frames=16, verbose=True):
        """
        Args:
            :verbose:    Setting verbose command

        Return:
            Does not return anything
        """
        self.verbose         = verbose
        self.alpha           = model_alpha
        self.input_alpha     = input_alpha
        self.resample_frames = resample_frames
        self.mean_clip       = np.load('models/c3d/crop_mean.npy')[...,::-1].tolist()
        self.name = 'c3d_alpha_exp'

        if verbose:
            print "C3D alpha Model Initialized"


    def _extraction_layer(self, inputs, params, model_input, L):
        """
        Args:
            :inputs:      Original inputs to the model
            :params:      Offset and sampling parameter estimates from Parameterization Network
            :model_input: Number of frames input into extraction layer
            :L:           Expected number of output frames

        Return:
            :output: Extracted features
        """
        max_alpha = 3.0
        min_alpha = 0.2

        sample_alpha_tick = tf.cond(params[0] > 0, lambda: tf.exp(-(params[0])), lambda: tf.exp(params[0]))
        # Scale alpha to be fixed in the range of 0.2 to 3.0
        sample_alpha_tick = sample_alpha_tick * (max_alpha-min_alpha) + min_alpha

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
        d1 = tf.reshape(tf.tile(d1, [112*112*3]), [L,112,112,3])
        d2 = tf.reshape(tf.tile(d2, [112*112*3]), [L,112,112,3])

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

        output = output - self.mean_clip

        return output

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

            layers['Parameterization_Variables'] = [tf.get_variable('alpha',shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.alpha))]

            layers['RAINlayer'] = tf.map_fn(lambda clip: self._extraction_layer(inputs=clip,
                                                       params=layers['Parameterization_Variables'],
                                                       model_input=input_dims, L=self.resample_frames), inputs)

            layers['conv1'] = conv3d_layer(input_tensor=layers['RAINlayer'],
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

    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step):
        """
        Args:
            :index:       Integer indicating the index of video frame from the text file containing video lists
            :data:        Data loaded from HDF5 files
            :labels:      Labels for loaded data
            :size:        List detailing values of height and width for final frames
            :is_training: Boolean value indication phase (TRAIN OR TEST)
        """
        return preprocess_tfrecords(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, self.input_alpha)


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
