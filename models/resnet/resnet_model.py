" RESNET-50 + LSTM MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "

import os

import tensorflow as tf
import numpy      as np

from models.models_abstract import Abstract_Model_Class
from utils.layers_utils     import *

from default_preprocessing    import preprocess
from video_preprocessing      import preprocess as preprocess_video
from video_full_preprocessing import preprocess as preprocess_video_full

class ResNet(Abstract_Model_Class):

    def __init__(self, **kwargs):
        """
        Args:
            Pass all arguments on to parent class, you may not add additional arguments without modifying abstract_model_class.py     and Models.py. Enter any additional initialization functionality here if desired.
        """
        super(ResNet, self).__init__(**kwargs)

    def _conv_block(self, n_filters, kernel_size, name, layer_numbers, input_layer, is_training, strides=2, weight_decay=0.0):
        """
        Args:
            :n_filters:     List detailing the number of filters
            :kernel_size:   List detailing the height and width of the kernel
            :name:          Name of the conv_block branch
            :layer_numbers: List detailing the connecting layer indices
            :input_layer:   Input layer to the conv_block
            :is_training:   Boolean variable indicating phase (TRAIN OR TEST)
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
                weight_decay = weight_decay, non_linear_fn=None,
                trainable=self.freeze)

        layers[layer_numbers[1]] = batch_normalization(layers[layer_numbers[0]],
                name='bn'+name+'_branch2a', training=is_training,
                trainable=self.freeze)

        layers[layer_numbers[2]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[1]]),
                filter_dims=[kernel_size, kernel_size, n_filters[1]], padding='SAME',
                name='res'+name+'_branch2b',
                weight_decay = weight_decay, non_linear_fn=None,
                trainable=self.freeze)

        layers[layer_numbers[3]] = batch_normalization(layers[layer_numbers[2]],
                name='bn'+name+'_branch2b', training=is_training, trainable=self.freeze)

        layers[layer_numbers[4]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[3]]),
                filter_dims=[1,1,n_filters[2]], padding = 'VALID',
                name='res'+name+'_branch2c',
                weight_decay = weight_decay, non_linear_fn=None,
                trainable=self.freeze)

        layers[layer_numbers[5]] = batch_normalization(layers[layer_numbers[4]],
                name='bn'+name+'_branch2c', training=is_training, trainable=self.freeze)

        # Shortcuts
        layers[layer_numbers[6]] = conv_layer(input_tensor=input_layer,
                filter_dims=[1,1,n_filters[2]], stride_dims=[strides, strides],
                padding = 'VALID',
                name='res'+name+'_branch1',
                weight_decay = weight_decay, non_linear_fn=None,
                trainable=self.freeze)

        layers[layer_numbers[7]] = batch_normalization(layers[layer_numbers[6]],
                name='bn'+name+'_branch1', training=is_training, trainable=self.freeze)

        # END OF CONV BLOCK

        layers[layer_numbers[8]] = tf.nn.relu(tf.add(layers[layer_numbers[5]],layers[layer_numbers[7]]))

        return layers


    def _identity_block(self, n_filters, kernel_size, name, layer_numbers, input_layer, is_training, weight_decay=0.0):
        """
        Args:
            :n_filters:     List detailing the number of filters
            :kernel_size:   List detailing the height and width of the kernel
            :name:          Name of the identity_block branch
            :layer_numbers: List detailing the connecting layer indices
            :input_layer:   Input layer to the identity_block
            :is_training:   Boolean variable indicating phase (TRAIN OR TEST)
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
                weight_decay = weight_decay, non_linear_fn=None,
                trainable=self.freeze)

        layers[layer_numbers[1]] = batch_normalization(layers[layer_numbers[0]],
                name='bn'+name+'_branch2a', training=is_training, trainable=self.freeze)

        layers[layer_numbers[2]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[1]]),
                filter_dims=[kernel_size, kernel_size, n_filters[1]], padding='SAME',
                name='res'+name+'_branch2b',
                weight_decay = weight_decay, non_linear_fn=None,
                trainable=self.freeze)

        layers[layer_numbers[3]] = batch_normalization(layers[layer_numbers[2]],
                name='bn'+name+'_branch2b', training=is_training, trainable=self.freeze)

        layers[layer_numbers[4]] = conv_layer(input_tensor=tf.nn.relu(layers[layer_numbers[3]]),
                filter_dims=[1,1,n_filters[2]], padding='VALID',
                name='res'+name+'_branch2c',
                weight_decay = weight_decay, non_linear_fn=None,
                trainable=self.freeze)

        layers[layer_numbers[5]] = batch_normalization(layers[layer_numbers[4]],
                name='bn'+name+'_branch2c', training=is_training, trainable=self.freeze)

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
            :return_layer: List of strings matching name of a layer in current model
            :weight_decay: Double value of weight decay

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """

        ############################################################################
        #                Creating ResNet+ LSTM Network Layers                   #
        ############################################################################

        if self.verbose:
            print('Generating ResNet network layers')

        # END IF

        inputs = inputs[0]

        with tf.name_scope(scope, 'resnet', [inputs]):
            layers = {}

            layers['1'] = conv_layer(input_tensor=inputs,
                    filter_dims=[7, 7, 64], stride_dims=[2,2],
                    padding = 'VALID',
                    name='conv1',
                    weight_decay = weight_decay, non_linear_fn=None,
                    trainable=self.freeze)

            layers['2'] = batch_normalization(layers['1'],
                    name='bn_conv1', training=False, trainable=self.freeze)

            layers['3'] = max_pool_layer(tf.nn.relu(layers['2']),
                                         filter_dims=[3, 3], stride_dims=[2,2],
                                         name='pool1', padding='VALID')

            layers.update(self._conv_block([64,64,256], kernel_size=3, name='2a', layer_numbers=['4','5','6','7','8','9','10','11','12'],
                            input_layer=layers['3'], is_training=False, strides=1))

            layers.update(self._identity_block([64,64,256], kernel_size=3, name='2b', layer_numbers=['13','14','15','16','17','18','19'],
                            input_layer=layers['12'], is_training=False))

            layers.update(self._identity_block([64,64,256], kernel_size=3, name='2c', layer_numbers=['20','21','22','23','24','25','26'],
                            input_layer=layers['19'], is_training=False))

            #########
            layers.update(self._conv_block([128,128,512], kernel_size=3, name='3a', layer_numbers=['27','28','29','30','31','32','33','34','35'],
                            input_layer=layers['26'], is_training=False))

            layers.update(self._identity_block([128,128,512], kernel_size=3, name='3b', layer_numbers=['36','37','38','39','40','41','42'],
                            input_layer=layers['35'], is_training=False))

            layers.update(self._identity_block([128,128,512], kernel_size=3, name='3c', layer_numbers=['43','44','45','46','47','48','49'],
                            input_layer=layers['42'], is_training=False))

            layers.update(self._identity_block([128,128,512], kernel_size=3, name='3d', layer_numbers=['50','51','52','53','54','55','56'],
                            input_layer=layers['49'], is_training=False))

            #########
            layers.update(self._conv_block([256,256,1024], kernel_size=3, name='4a', layer_numbers=['57','58','59','60','61','62','63','64','65'],
                            input_layer=layers['56'], is_training=False))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4b', layer_numbers=['66','67','68','69','70','71','72'],
                            input_layer=layers['65'], is_training=False))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4c', layer_numbers=['73','74','75','76','77','78','79'],
                            input_layer=layers['72'], is_training=False))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4d', layer_numbers=['80','81','82','83','84','85','86'],
                            input_layer=layers['79'], is_training=False))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4e', layer_numbers=['87','88','89','90','91','92','93'],
                            input_layer=layers['86'], is_training=False))

            layers.update(self._identity_block([256,256,1024], kernel_size=3, name='4f', layer_numbers=['94','95','96','97','98','99','100'],
                            input_layer=layers['93'], is_training=False))

            #########
            layers.update(self._conv_block([512,512,2048], kernel_size=3, name='5a', layer_numbers=['101','102','103','104','105','106','107','108','109'],
                            input_layer=layers['100'], is_training=False))

            layers.update(self._identity_block([512,512,2048], kernel_size=3, name='5b', layer_numbers=['110','111','112','113','114','115','116'],
                            input_layer=layers['109'], is_training=False))

            layers.update(self._identity_block([512,512,2048], kernel_size=3, name='5c', layer_numbers=['117','118','119','120','121','122','123'],
                            input_layer=layers['116'], is_training=False))

            layers['124'] = tf.reduce_mean(layers['123'], reduction_indices=[1,2], name='avg_pool')

            layers['125'] = lstm(layers['124'], seq_length, feat_size=2048, cell_size=512)

            layers['126'] = dropout(layers['125'], training=is_training, rate=dropout_rate)

            layers['logits'] = tf.expand_dims(fully_connected_layer(input_tensor=layers['126'], out_dim=output_dims, non_linear_fn=None, name='logits', weight_decay=weight_decay), 0)

        # END WITH

        return [layers[x] for x in return_layer]

    def load_default_weights(self):
        """
        return: Numpy dictionary containing the names and values of the weight tensors used to initialize this model
        """
        return np.load('models/weights/resnet50_rgb_imagenet.npy')

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

        Return:
            Pointer to preprocessing function of current model
        """
        if self.preproc_method == 'video':
            return preprocess_video(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, self.input_alpha, istraining)

        elif self.preproc_method == 'video_full':
            return preprocess_video_full(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, self.input_alpha, istraining)

        else:
            return preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, self.input_alpha, istraining)

    """ Function to return loss calculated on half the outputs of a given network """
    def half_loss(self, logits, labels):
        """
        Args:
            :logits: Unscaled logits returned from final layer in model
            :labels: True labels corresponding to loaded data

        Return:
            Cross entropy loss value
        """

        labels = tf.cast(labels, tf.int64)

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels[:,:labels.shape[1].value/2],
                                                                  logits=logits[:,:logits.shape[1].value/2,:])
        return cross_entropy_loss

    """ Function to return loss calculated on all the outputs of a given network """
    def full_loss(self, logits, labels):
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

    """ Function to return loss calculated on given network """
    def loss(self, logits, labels, loss_type='full_loss'):
        """
        Args:
            :logits: Unscaled logits returned from final layer in model
            :labels: True labels corresponding to loaded data

        Return:
            Cross entropy loss value
        """
        if loss_type == 'full_loss':
            return self.full_loss(logits, labels)

        else:
            return self.half_loss(logits, labels)

        # END IF
