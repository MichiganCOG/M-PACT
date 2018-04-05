" TSN MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "

import tensorflow as tf
import numpy      as np

from models.abstract_model_class import Abstract_Model_Class
from utils.layers_utils          import *

from default_preprocessing       import preprocess


class TSN(Abstract_Model_Class):

    def __init__(self, **kwargs):
        """
        Args:
            Pass all arguments on to parent class, you may not add additional arguments without modifying abstract_model_class.py and Models.py. Enter any additional initialization functionality here if desired.
        """
        super(TSN, self).__init__(**kwargs)


    def _inception_block_with_pool(self, inputs, filter_list, pool_type='avg', scope='', weight_decay=0.0):
        layers = {}

        layers[scope+'_1'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[0]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/1x1')
        layers[scope+'_1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_1'], name=scope+'/1x1_bn'))

        layers[scope+'_2_reduce'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[1]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/3x3_reduce')
        layers[scope+'_2_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_2_reduce'], name=scope+'/3x3_reduce_bn'))
        layers[scope+'_2'] = conv_layer(input_tensor=layers[scope+'_2_reduce_bn'], filter_dims=[3,3,filter_list[2]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/3x3')
        layers[scope+'_2_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_2'], name=scope+'/3x3_bn'))

        layers[scope+'_double_reduce'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[3]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_reduce')
        layers[scope+'_double_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_reduce'], name=scope+'/double_3x3_reduce_bn'))
        layers[scope+'_double_1'] = conv_layer(input_tensor=layers[scope+'_double_reduce_bn'], filter_dims=[3,3,filter_list[4]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_1')
        layers[scope+'_double_1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_1'], name=scope+'/double_3x3_1_bn'))
        layers[scope+'_double_2'] = conv_layer(input_tensor=layers[scope+'_double_1_bn'], filter_dims=[3,3,filter_list[5]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_2')
        layers[scope+'_double_2_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_2'], name=scope+'/double_3x3_2_bn'))

        if pool_type=='avg':
            layers[scope+'_pool'] = avg_pool_layer(input_tensor=pad(inputs, 1), filter_dims=[3,3], stride_dims=[1,1], padding='VALID', name=scope+'/pool')

        else:
            layers[scope+'_pool'] = max_pool_layer(input_tensor=pad(inputs, 1), filter_dims=[3,3], stride_dims=[1,1], padding='VALID', name=scope+'/pool')

        # END IF

        layers[scope+'_pool_proj'] = conv_layer(input_tensor=layers[scope+'_pool'], filter_dims=[1,1,filter_list[6]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/pool_proj')
        layers[scope+'_pool_proj_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_pool_proj'], name=scope+'/pool_proj_bn'))
        layers[scope+'_output'] = tf.concat([layers[scope+'_1_bn'] + layers[scope+'_2_bn'] + layers[scope+'_double_bn'] + layers[scope+'_pool_proj_bn']], axis=3, name=scope+'/output')

        return layers

    def _inception_block_no_pool(self, inputs, filter_list, scope='', weight_decay=0.0):
        layers = {}

        layers[scope+'_1_reduce'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[1]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/3x3_reduce')
        layers[scope+'_1_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_1_reduce'], name=scope+'/3x3_reduce_bn'))
        layers[scope+'_1'] = conv_layer(input_tensor=layers[scope+'_1_reduce_bn'], filter_dims=[3,3,filter_list[2]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/3x3')
        layers[scope+'_1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_1'], name=scope+'/3x3_bn'))

        layers[scope+'_double_reduce'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[3]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_reduce')
        layers[scope+'_double_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_reduce'], name=scope+'/double_3x3_reduce_bn'))
        layers[scope+'_double_1'] = conv_layer(input_tensor=layers[scope+'_double_reduce_bn'], filter_dims=[3,3,filter_list[4]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_1')
        layers[scope+'_double_1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_1'], name=scope+'/double_3x3_1_bn'))
        layers[scope+'_double_2'] = conv_layer(input_tensor=layers[scope+'_double_1_bn'], filter_dims=[3,3,filter_list[5]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_2')
        layers[scope+'_double_2_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_2'], name=scope+'/double_3x3_2_bn'))

        layers[scope+'_pool'] = max_pool_layer(input_tensor=pad(inputs, 1), filter_dims=[3,3], stride_dims=[2,2], padding='SAME', name=scope+'/pool')

        layers[scope+'_output'] = tf.concat([layers[scope+'_1_bn'] + layers[scope+'_double_bn'] + layers[scope+'_pool']], axis=3, name=scope+'/output')

        return layers

    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, batch_size, scope, dropout_rate = 0.5, return_layer=['logits'], weight_decay=0.0):
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
            :batch_size:   Number of videos or clips to process at a time

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """

        ############################################################################
        #                       Add TSN Network Layers HERE                  #
        ############################################################################

        if self.verbose:
            print('Generating TSN network layers')

        # END IF

        with tf.name_scope(scope, 'TSN', [inputs]):
            layers = {}

            layers['conv1'] = conv_layer(input_tensor=inputs, filter_dims=[7,7,64], stride_dims=[2,2], non_linear_fn=None, name='conv1/7x7_s2', weight_decay=weight_decay)

            layers['conv1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers['conv1'], name='conv1/7x7_s2_bn'))

            layers['pool1'] = max_pool_layer(input_tensor=layers['conv1_bn'], filter_dims=[3,3], stride_dims=[2,2], name='pool1/3x3_s2')


            layers['conv2_reduce'] = conv_layer(input_tensor=layers['pool1'], filter_dims=[1,1,64], stride_dims=[1,1], non_linear_fn=None, name='conv2/3x3_reduce', weight_decay=weight_decay)

            layers['conv2_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers['conv2_reduce'], name='conv2/3x3_reduce_bn'))

            layers['conv2'] = conv_layer(input_tensor=layers['conv2_reduce_bn'], filter_dims=[3,3,64], stride_dims=[1,1], non_linear_fn=None, name='conv2/3x3', weight_decay=weight_decay)

            layers['conv2_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers['conv2'], name='conv2/3x3_bn'))

            layers['pool2'] = max_pool_layer(input_tensor=layers['conv2_bn'], filter_dims=[3,3], stride_dims=[2,2], name='pool2/3x3_s2')




            ########################################################################################
            #        TODO: Add any desired layers from layers_utils to this layers dictionary      #
            #                                                                                      #
            #       EX: layers['conv1'] = conv3d_layer(input_tensor=inputs,                        #
            #           filter_dims=[dim1, dim2, dim3, dim4],                                      #
            #           name=NAME,                                                                 #
            #           weight_decay = wd)                                                         #
            ########################################################################################


            ########################################################################################
            #       TODO: Final Layer must be 'logits'                                             #
            #                                                                                      #
            #  EX:  layers['logits'] = [fully_connected_layer(input_tensor=layers['previous'],     #
            #                                         out_dim=output_dims, non_linear_fn=None,     #
            #                                         name='out', weight_decay=weight_decay)]      #
            ########################################################################################

            layers['logits'] = # TODO Every model must return a layer named 'logits'

            layers['logits'] = tf.reshape(layers['logits'], [batch_size, seq_length, output_dims])

        # END WITH

        return [layers[x] for x in return_layer]




#    def load_default_weights(self):
#        """
#        return: Numpy dictionary containing the names and values of the weight tensors used to initialize this model
#        """
#
#        ############################################################################
#        # TODO: Add default model weights to models/weights/ and import them here  #
#        #                          ( OPTIONAL )                                    #
#        #                                                                          #
#        # EX: return np.load('models/weights/model_weights.npy')                   #
#        #                                                                          #
#        ############################################################################




    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step):
        """
        Args:
            :input_data_tensor:     Data loaded from tfrecords containing either video or clips
            :frames:                Number of frames in loaded video or clip
            :height:                Pixel height of loaded video or clip
            :width:                 Pixel width of loaded video or clip
            :channel:               Number of channels in video or clip, usually 3 (RGB)
            :input_dims:            Number of frames used in input
            :output_dims:           Integer number of classes in current dataset
            :seq_length:            Length of output sequence
            :size:                  List detailing values of height and width for final frames
            :label:                 Label for loaded data
            :is_training:           Boolean value indication phase (TRAIN OR TEST)
            :video_step:            Tensorflow variable indicating the total number of videos (not clips) that have been loaded
        """

        ####################################################
        # TODO: Add more preprcessing arguments if desired #
        ####################################################

        return preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step, self.input_alpha)



   """ Function to return loss calculated on given network """
   def loss(self, logits, labels, loss_type):
       """
       Args:
           :logits:     Unscaled logits returned from final layer in model
           :labels:     True labels corresponding to loaded data
           :loss_type:  Allow for multiple losses that can be selected at run time. Implemented through if statements
       """

       ####################################################################################
       #  TODO: ADD CUSTOM LOSS HERE, DEFAULT IS CROSS ENTROPY LOSS                       #
       #                                                                                  #
       #   EX: labels = tf.cast(labels, tf.int64)                                         #
       #       cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, #
       #                                                            logits=logits)        #
       #        return cross_entropy_loss                                                 #
       ####################################################################################
