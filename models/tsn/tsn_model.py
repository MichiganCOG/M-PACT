" TSN MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "

import tensorflow as tf
import numpy      as np

# Try imports assuming regular framework structure
try:
    from models.models_abstract      import Abstract_Model_Class
    from utils.layers_utils          import *

# Try imports assuming call is coming from models/current_model/current_model.py
except:
    import sys
    sys.path.append('../../')
    from models.models_abstract import Abstract_Model_Class
    from utils.layers_utils     import *

# END TRY

from default_preprocessing       import preprocess

class TSN(Abstract_Model_Class):

    def __init__(self, **kwargs):
        """
        Args:
            Pass all arguments on to parent class, you may not add additional arguments without modifying abstract_model_class.py, Models.py, train.py, and test.py. Enter any additional initialization functionality here if desired.
        """
        super(TSN, self).__init__(**kwargs)

        self.num_segs = 3


    def _inception_block_with_pool(self, inputs, filter_list, is_training, pool_type='avg', scope='', weight_decay=0.0):
        layers = {}

        layers[scope+'_1'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[0]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/1x1')
        layers[scope+'_1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_1'], training=False, trainable=False, name=scope+'/1x1_bn'))

        layers[scope+'_2_reduce'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[1]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/3x3_reduce')
        layers[scope+'_2_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_2_reduce'], training=False, trainable=False, name=scope+'/3x3_reduce_bn'))
        layers[scope+'_2'] = conv_layer(input_tensor=pad(layers[scope+'_2_reduce_bn'], 1), filter_dims=[3,3,filter_list[2]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/3x3')
        layers[scope+'_2_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_2'], training=False, trainable=False, name=scope+'/3x3_bn'))

        layers[scope+'_double_reduce'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[3]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_reduce')
        layers[scope+'_double_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_reduce'], training=False, trainable=False, name=scope+'/double_3x3_reduce_bn'))
        layers[scope+'_double_1'] = conv_layer(input_tensor=pad(layers[scope+'_double_reduce_bn'], 1), filter_dims=[3,3,filter_list[4]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_1')
        layers[scope+'_double_1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_1'], training=False, trainable=False, name=scope+'/double_3x3_1_bn'))
        layers[scope+'_double_2'] = conv_layer(input_tensor=pad(layers[scope+'_double_1_bn'], 1), filter_dims=[3,3,filter_list[5]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_2')
        layers[scope+'_double_2_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_2'], training=False, trainable=False, name=scope+'/double_3x3_2_bn'))

        if pool_type=='avg':
            layers[scope+'_pool'] = avg_pool_layer(input_tensor=pad(inputs, 1), filter_dims=[3,3], stride_dims=[1,1], padding='VALID', name=scope+'/pool')

        else:
            layers[scope+'_pool'] = max_pool_layer(input_tensor=pad(inputs, 1), filter_dims=[3,3], stride_dims=[1,1], padding='VALID', name=scope+'/pool')

        # END IF

        layers[scope+'_pool_proj'] = conv_layer(input_tensor=layers[scope+'_pool'], filter_dims=[1,1,filter_list[6]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/pool_proj')
        layers[scope+'_pool_proj_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_pool_proj'], training=False, trainable=False, name=scope+'/pool_proj_bn'))

        layers[scope+'_output'] = tf.concat([layers[scope+'_1_bn'], layers[scope+'_2_bn'], layers[scope+'_double_2_bn'], layers[scope+'_pool_proj_bn']], axis=3, name=scope+'/output')

        return layers

    def _inception_block_no_pool(self, inputs, filter_list, is_training, scope='', weight_decay=0.0):
        layers = {}

        layers[scope+'_1_reduce'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[0]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/3x3_reduce')
        layers[scope+'_1_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_1_reduce'], training=False, trainable=False, name=scope+'/3x3_reduce_bn'))
        layers[scope+'_1'] = conv_layer(input_tensor=pad(layers[scope+'_1_reduce_bn'], 1), filter_dims=[3,3,filter_list[1]], stride_dims=[2,2], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/3x3')
        layers[scope+'_1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_1'], training=False, trainable=False, name=scope+'/3x3_bn'))

        layers[scope+'_double_reduce'] = conv_layer(input_tensor=inputs, filter_dims=[1,1,filter_list[2]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_reduce')
        layers[scope+'_double_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_reduce'], training=False, trainable=False, name=scope+'/double_3x3_reduce_bn'))
        layers[scope+'_double_1'] = conv_layer(input_tensor=pad(layers[scope+'_double_reduce_bn'], 1), filter_dims=[3,3,filter_list[3]], stride_dims=[1,1], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_1')
        layers[scope+'_double_1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_1'], training=False, trainable=False, name=scope+'/double_3x3_1_bn'))
        layers[scope+'_double_2'] = conv_layer(input_tensor=pad(layers[scope+'_double_1_bn'], 1), filter_dims=[3,3,filter_list[4]], stride_dims=[2,2], non_linear_fn=None, padding='VALID', weight_decay=weight_decay, name=scope+'/double_3x3_2')
        layers[scope+'_double_2_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers[scope+'_double_2'], training=False, trainable=False, name=scope+'/double_3x3_2_bn'))

        layers[scope+'_pool'] = max_pool_layer(input_tensor=inputs, filter_dims=[3,3], stride_dims=[2,2], padding='SAME', name=scope+'/pool')

        layers[scope+'_output'] = tf.concat([layers[scope+'_1_bn'], layers[scope+'_double_2_bn'], layers[scope+'_pool']], axis=3, name=scope+'/output')

        return layers


    def flatten_batch(self, inputs):
        in_dims = inputs.get_shape().as_list()
        out_dims = [in_dims[0]*in_dims[1]]
        out_dims.extend(in_dims[2:])
        out_dims = np.array(out_dims)
        out_dims[out_dims==None] = -1
        return tf.reshape(inputs, out_dims)

    def extend_batch(self, inputs, batch_size):
        in_dims = inputs.get_shape().as_list()
        out_dims = [batch_size, in_dims[0]/batch_size]
        out_dims.extend(in_dims[1:])
        out_dims = np.array(out_dims)
        out_dims[out_dims==None] = -1
        return tf.reshape(inputs, out_dims)


    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, dropout_rate = 0.2, return_layer=['logits'], weight_decay=0.0):
        """
        Args:
            :inputs:       Input to model of shape [BatchSize x Frames x Height x Width x Channels]
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
        #                               TSN Network Layers                         #
        ############################################################################

        if self.verbose:
            print('Generating TSN network layers')

        # END IF

        inputs = self.flatten_batch(inputs)

        with tf.name_scope(scope, 'TSN', [inputs]):
            layers = {}

            layers['conv1'] = conv_layer(input_tensor=inputs, filter_dims=[7,7,64], stride_dims=[2,2], non_linear_fn=None, name='conv1/7x7_s2', weight_decay=weight_decay)
            layers['conv1_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers['conv1'], training=is_training, trainable=False, name='conv1/7x7_s2_bn'))
            layers['pool1'] = max_pool_layer(input_tensor=layers['conv1_bn'], filter_dims=[3,3], stride_dims=[2,2], name='pool1/3x3_s2')

            layers['conv2_reduce'] = conv_layer(input_tensor=layers['pool1'], filter_dims=[1,1,64], stride_dims=[1,1], non_linear_fn=None, name='conv2/3x3_reduce', weight_decay=weight_decay)
            layers['conv2_reduce_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers['conv2_reduce'], training=False, trainable=False, name='conv2/3x3_reduce_bn'))
            layers['conv2'] = conv_layer(input_tensor=layers['conv2_reduce_bn'], filter_dims=[3,3,192], stride_dims=[1,1], non_linear_fn=None, name='conv2/3x3', weight_decay=weight_decay)
            layers['conv2_bn'] = tf.nn.relu(batch_normalization(input_tensor=layers['conv2'], training=False, trainable=False, name='conv2/3x3_bn'))
            layers['pool2'] = max_pool_layer(input_tensor=layers['conv2_bn'], filter_dims=[3,3], stride_dims=[2,2], name='pool2/3x3_s2')

            layers.update(self._inception_block_with_pool(inputs=layers['pool2'], filter_list=[64, 64, 64, 64, 96, 96, 32], pool_type='avg', scope='inception_3a', is_training=is_training, weight_decay=weight_decay))
            layers.update(self._inception_block_with_pool(inputs=layers['inception_3a_output'], filter_list=[64, 64, 96, 64, 96, 96, 64], pool_type='avg', scope='inception_3b', is_training=is_training, weight_decay=weight_decay))
            layers.update(self._inception_block_no_pool(inputs=layers['inception_3b_output'], filter_list=[128, 160, 64, 96, 96], scope='inception_3c', is_training=is_training, weight_decay=weight_decay))

            layers.update(self._inception_block_with_pool(inputs=layers['inception_3c_output'], filter_list=[224, 64, 96, 96, 128, 128, 128], pool_type='avg', scope='inception_4a', is_training=is_training, weight_decay=weight_decay))
            layers.update(self._inception_block_with_pool(inputs=layers['inception_4a_output'], filter_list=[192, 96, 128, 96, 128, 128, 128], pool_type='avg', scope='inception_4b', is_training=is_training, weight_decay=weight_decay))
            layers.update(self._inception_block_with_pool(inputs=layers['inception_4b_output'], filter_list=[160, 128, 160, 128, 160, 160, 128], pool_type='avg', scope='inception_4c', is_training=is_training, weight_decay=weight_decay))
            layers.update(self._inception_block_with_pool(inputs=layers['inception_4c_output'], filter_list=[96, 128, 192, 160, 192, 192, 128], pool_type='avg', scope='inception_4d', is_training=is_training, weight_decay=weight_decay))
            layers.update(self._inception_block_no_pool(inputs=layers['inception_4d_output'], filter_list=[128, 192, 192, 256, 256], scope='inception_4e', is_training=is_training, weight_decay=weight_decay))

            layers.update(self._inception_block_with_pool(inputs=layers['inception_4e_output'], filter_list=[352, 192, 320, 160, 224, 224, 128], pool_type='avg', scope='inception_5a', is_training=is_training, weight_decay=weight_decay))
            layers.update(self._inception_block_with_pool(inputs=layers['inception_5a_output'], filter_list=[352, 192, 320, 192, 224, 224, 128], pool_type='max', scope='inception_5b', is_training=is_training, weight_decay=weight_decay))

            layers['global_pool'] = avg_pool_layer(input_tensor=layers['inception_5b_output'], filter_dims=[7,7], stride_dims=[1,1], name='global_pool', padding='VALID')

            layers['dropout'] = dropout(input_tensor=layers['global_pool'], training=is_training, rate=dropout_rate)

            layers['logits'] = fully_connected_layer(input_tensor=layers['dropout'], out_dim=output_dims, name='fc-action', non_linear_fn=None, weight_init=tf.truncated_normal_initializer(stddev=0.001), weight_decay=weight_decay)

        # END WITH

        layers['logits'] = self.extend_batch(layers['logits'], self.batch_size)

        return [layers[x] for x in return_layer]




    def load_default_weights(self):
       """
       return: Numpy dictionary containing the names and values of the weight tensors used to initialize this model
       """

       return np.load('models/weights/tsn_BNInception_ImageNet_pretrained.npy')#tsn_pretrained_UCF101_reordered.npy')#bn_inception_rgb_init.npy')



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
        return preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step, self.num_segs, self.input_alpha)



    """ Function to return loss calculated on given network """
    def loss(self, logits, labels, loss_type):
        """
        Args:
           :logits:     Unscaled logits returned from final layer in model [batchSize x seqLength]
           :labels:     True labels corresponding to loaded data [batchSize x seqLength x outputDims]
           :loss_type:  Allow for multiple losses that can be selected at run time. Implemented through if statements
        """

        labels = tf.cast(labels, tf.int64)

        labels = tf.reduce_mean(labels, 1)
        logits = tf.reduce_mean(logits, 1)

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                  logits=logits)
        return cross_entropy_loss


# Base testing setup to check if model loads
if __name__=="__main__":
    batch_size = 128
    x = tf.placeholder(tf.float32, shape=(batch_size, 3, 224, 224, 3))
    y = tf.placeholder(tf.int32, [51])
    network = TSN(modelName='tsn', inputDims=3, outputDims=51, expName='tsn_test', numVids=3570, batchSize=batch_size, preprocMethod='default', clipLength=-1, numEpochs=1, numClips=-1, numGpus=1, train=1, modelAlpha=1, inputAlpha=1, dropoutRate=0.8, freeze=0, verbose=1)
    XX =  network.inference(x, is_training=True, input_dims=3, output_dims=51, seq_length=3, scope='my_scope')
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    import pdb; pdb.set_trace()
