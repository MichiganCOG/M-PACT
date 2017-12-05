# TSN Model implemnetation for use with tensorflow


# Import tensorflow modules
import tensorflow as tf

from tsn_preprocessing_TFRecords import preprocess as preprocess_tfrecords

# Import math and numpy modules
import numpy as np

# Import HDF5
import h5py

import os

import sys
sys.path.append('../..')
from layers_utils import *


class TSN():

    def __init__(self, verbose=True):
        self.verbose=verbose
        self.name = 'tsn'


    def _main_block(self, nFilters, name, inputLayer, dataDict, poolType = 'AVG', weight_decay=0.0):

        layers = {}

        layers[name+'_1'] = conv_layer(input_tensor=inputLayer,
                       filter_dims=[1,1,nFilters[0]],
                       stride_dims=[1,1],
                       name=name+'_1',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['1x1_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['1x1_1']))

        layers[name+'_1_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_1'],
                name=name+'_1_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['1x1_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['1x1_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['1x1_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['1x1_bn_0'])))


        layers[name+'_2_reduce'] = conv_layer(input_tensor=inputLayer,
                       filter_dims=[1,1,nFilters[1]],
                       stride_dims=[1,1],
                       name=name+'_2_reduce',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['3x3_reduce_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['3x3_reduce_1']))

        layers[name+'_2_reduce_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_2_reduce'],
                name=name+'_2_reduce_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['3x3_reduce_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['3x3_reduce_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['3x3_reduce_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['3x3_reduce_bn_0'])))


        layers[name+'_2'] = conv_layer(input_tensor=pad(layers[name+'_2_reduce_bn'], 1),
                       filter_dims=[3,3,nFilters[2]],
                       stride_dims=[1,1],
                       name=name+'_2',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['3x3_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['3x3_1']))

        layers[name+'_2_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_2'],
                name=name+'_2_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['3x3_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['3x3_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['3x3_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['3x3_bn_0'])))


        layers[name+'_double_reduce'] = conv_layer(input_tensor=inputLayer,
                       filter_dims=[1,1,nFilters[3]],
                       stride_dims=[1,1],
                       name=name+'_double_reduce',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['double_3x3_reduce_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['double_3x3_reduce_1']))

        layers[name+'_double_reduce_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_double_reduce'],
                name=name+'_double_reduce_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['double_3x3_reduce_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['double_3x3_reduce_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['double_3x3_reduce_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['double_3x3_reduce_bn_0'])))


        layers[name+'_double_1'] = conv_layer(input_tensor=pad(layers[name+'_double_reduce_bn'], 1),
                       filter_dims=[3,3,nFilters[4]],
                       stride_dims=[1,1],
                       name=name+'_double_1',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['double_3x3_1_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['double_3x3_1_1']))

        layers[name+'_double_1_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_double_1'],
                name=name+'_double_1_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['double_3x3_1_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['double_3x3_1_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['double_3x3_1_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['double_3x3_1_bn_0'])))


        layers[name+'_double_2'] = conv_layer(input_tensor=pad(layers[name+'_double_1_bn'], 1),
                       filter_dims=[3,3,nFilters[5]],
                       stride_dims=[1,1],
                       name=name+'_double_2',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['double_3x3_2_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['double_3x3_2_1']))

        layers[name+'_double_2_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_double_2'],
                name=name+'_double_2_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['double_3x3_2_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['double_3x3_2_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['double_3x3_2_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['double_3x3_2_bn_0'])))


        if poolType == "AVG":
            layers[name+'_pool'] = avg_pool_layer(input_tensor=pad(inputLayer, 1), filter_dims=[3,3],
                            stride_dims=[1,1], name=name+'_pool', padding='VALID')
        else:
            layers[name+'_pool'] = max_pool_layer(input_tensor=pad(inputLayer, 1), filter_dims=[3,3],
                            stride_dims=[1,1], name=name+'_pool', padding='VALID')


        layers[name+'_pool_proj'] = conv_layer(input_tensor=layers[name+'_pool'],
                       filter_dims=[1,1,nFilters[6]],
                       stride_dims=[1,1],
                       name=name+'_pool_proj',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['pool_proj_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['pool_proj_1']))

        layers[name+'_pool_proj_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_pool_proj'],
                name=name+'_pool_proj_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['pool_proj_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['pool_proj_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['pool_proj_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['pool_proj_bn_0'])))

        # Axis 3 for concatenation along channels
        layers[name+'_output'] = tf.concat([layers[name+'_1_bn'], layers[name+'_2_bn'], layers[name+'_double_2_bn'], layers[name+'_pool_proj_bn']], axis=3)

        return layers















    def _end_block(self, nFilters, name, inputLayer, dataDict, weight_decay=0.0):

        layers = {}
        print name
        #import pdb;pdb.set_trace()
        layers[name+'_1_reduce'] = conv_layer(input_tensor=inputLayer,
                       filter_dims=[1,1,nFilters[0]],
                       stride_dims=[1,1],
                       name=name+'_1_reduce',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['3x3_reduce_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['3x3_reduce_1']))

        layers[name+'_1_reduce_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_1_reduce'],
                name=name+'_1_reduce_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['3x3_reduce_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['3x3_reduce_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['3x3_reduce_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['3x3_reduce_bn_0'])))


        layers[name+'_1'] = conv_layer(input_tensor=pad(layers[name+'_1_reduce_bn'], 1),
                       filter_dims=[3,3,nFilters[1]],
                       stride_dims=[2,2],
                       name=name+'_1',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['3x3_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['3x3_1']))

        layers[name+'_1_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_1'],
                name=name+'_1_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['3x3_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['3x3_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['3x3_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['3x3_bn_0'])))


        layers[name+'_double_reduce'] = conv_layer(input_tensor=inputLayer,
                       filter_dims=[1,1,nFilters[2]],
                       stride_dims=[1,1],
                       name=name+'_double_reduce',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['double_3x3_reduce_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['double_3x3_reduce_1']))

        layers[name+'_double_reduce_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_double_reduce'],
                name=name+'_double_reduce_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['double_3x3_reduce_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['double_3x3_reduce_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['double_3x3_reduce_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['double_3x3_reduce_bn_0'])))


        layers[name+'_double_1'] = conv_layer(input_tensor=pad(layers[name+'_double_reduce_bn'], 1),
                       filter_dims=[3,3,nFilters[3]],
                       stride_dims=[1,1],
                       name=name+'_double_1',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['double_3x3_1_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['double_3x3_1_1']))

        layers[name+'_double_1_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_double_1'],
                name=name+'_double_1_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['double_3x3_1_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['double_3x3_1_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['double_3x3_1_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['double_3x3_1_bn_0'])))


        layers[name+'_double_2'] = conv_layer(input_tensor=pad(layers[name+'_double_1_bn'], 1),
                       filter_dims=[3,3,nFilters[4]],
                       stride_dims=[2,2],
                       name=name+'_double_2',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict[name]['double_3x3_2_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict[name]['double_3x3_2_1']))

        layers[name+'_double_2_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers[name+'_double_2'],
                name=name+'_double_2_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict[name]['double_3x3_2_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict[name]['double_3x3_2_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict[name]['double_3x3_2_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict[name]['double_3x3_2_bn_0'])))

        #import pdb;pdb.set_trace()

        layers[name+'_pool'] = max_pool_layer(input_tensor=inputLayer, filter_dims=[3,3], stride_dims=[2,2], name=name+'_pool',padding='SAME')


        # Axis 3 for concatenation along channels
        layers[name+'_output'] = tf.concat([layers[name+'_1_bn'], layers[name+'_double_2_bn'], layers[name+'_pool']], axis=3)

        return layers









#max_pool_layer(input_tensor=inputLayer, filter_dims=[3,3], stride_dims=[2,2], name=name+'_pool',padding='VALID')







    def inference(self, inputs, isTraining, inputDims, outputDims, seqLength, scope, k, j, return_layer='logits', dataDict=None, cpuId = 0, weight_decay=0.0):

        ############################################################################
        #                         Creating TSN Network Layers                      #
        ############################################################################

        print('Generating TSN network layers')

        if outputDims == 51:
            dataDict = np.load('/z/home/erichof/TF_Activity_Recognition_Framework/models/tsn/tsn_HMDB51.npy').tolist()
    #    dataDict = np.load('/z/home/madantrg/RILCode/Code_ND/Utils/vgg16.npy').item()
        else:
            dataDict = np.load('/z/home/erichof/TF_Activity_Recognition_Framework/models/tsn/tsn_UCF101.npy').tolist()

        if isTraining:
            keep_prob = 0.2
        else:
            keep_prob = 1.0


        layers = {}

        ################################### conv1  ######################################


        layers['conv1'] = conv_layer(input_tensor=pad(inputs, 3),
                       filter_dims=[7,7,64],
                       stride_dims=[2,2],
                       name='conv1',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict['conv1']['7x7_s2_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict['conv1']['7x7_s2_1']))

        layers['conv1_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers['conv1'],
                name='conv1_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict['conv1']['7x7_s2_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict['conv1']['7x7_s2_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict['conv1']['7x7_s2_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict['conv1']['7x7_s2_bn_0']),
                training=False,
                momentum=0.1))


        layers['pool1'] = max_pool_layer(layers['conv1_bn'],
                           filter_dims=[3,3],
                           stride_dims=[2,2],
                           name='pool1',
                           padding='SAME')



        ################################### conv2  ######################################

        layers['conv2_reduce'] = conv_layer(input_tensor=layers['pool1'],
                       filter_dims=[1,1,64],
                       stride_dims=[1,1],
                       name='conv2_reduce',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict['conv2']['3x3_reduce_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict['conv2']['3x3_reduce_1']))

        layers['conv2_reduce_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers['conv2_reduce'],
                name='conv2_reduce_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict['conv2']['3x3_reduce_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict['conv2']['3x3_reduce_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict['conv2']['3x3_reduce_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict['conv2']['3x3_reduce_bn_0']),
                training=False))



        layers['conv2'] = conv_layer(input_tensor=pad(layers['conv2_reduce_bn'], 1),
                       filter_dims=[3,3,192],
                       stride_dims=[1,1],
                       name='conv2',
                       weight_decay=weight_decay,
                       padding='VALID',
                       non_linear_fn=None,
                       kernel_init=tf.constant_initializer(dataDict['conv2']['3x3_0'].transpose((2,3,1,0))),
                       bias_init=tf.constant_initializer(dataDict['conv2']['3x3_1']))

        layers['conv2_bn'] = tf.nn.relu(tf.layers.batch_normalization(layers['conv2'],
                name='conv2_bn',
                moving_mean_initializer=tf.constant_initializer(dataDict['conv2']['3x3_bn_2']),
                moving_variance_initializer=tf.constant_initializer(dataDict['conv2']['3x3_bn_3']),
                beta_initializer=tf.constant_initializer(dataDict['conv2']['3x3_bn_1']),
                gamma_initializer=tf.constant_initializer(dataDict['conv2']['3x3_bn_0'])))

        layers['pool2'] = max_pool_layer(layers['conv2_bn'],
                           filter_dims=[3,3],
                           stride_dims=[2,2],
                           name='pool2',
                           padding='SAME')

        ################################### inception_3a  ######################################



        layers.update(self._main_block([64,64,64,64,96,96,32], 'inception_3a', layers['pool2'],
                        dataDict, poolType = 'AVG', weight_decay=weight_decay))



        ################################### inception_3b  ######################################

        layers.update(self._main_block([64,64,96,64,96,96,64], 'inception_3b', layers['inception_3a_output'],
                        dataDict, poolType = 'AVG', weight_decay=weight_decay))




        ################################### inception_3c  ######################################
    #    import pdb; pdb.set_trace()
        layers.update(self._end_block([128,160,64,96,96], 'inception_3c', layers['inception_3b_output'],
                        dataDict, weight_decay=weight_decay))

        ################################### inception_4a  ######################################

        layers.update(self._main_block([224,64,96,96,128,128,128], 'inception_4a', layers['inception_3c_output'],
                        dataDict, poolType = 'AVG', weight_decay=weight_decay))

        ################################### inception_4b  ######################################
        layers.update(self._main_block([192,96,128,96,128,128,128], 'inception_4b', layers['inception_4a_output'],
                        dataDict, poolType = 'AVG', weight_decay=weight_decay))


        ################################### inception_4c  ######################################
        layers.update(self._main_block([160,128,160,128,160,160,128], 'inception_4c', layers['inception_4b_output'],
                        dataDict, poolType = 'AVG', weight_decay=weight_decay))

        ################################### inception_4d  ######################################
        layers.update(self._main_block([96,128,192,160,192,192,128], 'inception_4d', layers['inception_4c_output'],
                        dataDict, poolType = 'AVG', weight_decay=weight_decay))


        ################################### inception_4e  ######################################
        layers.update(self._end_block([128,192,192,256,256], 'inception_4e', layers['inception_4d_output'],
                        dataDict, weight_decay=weight_decay))


        ################################### inception_5a  ######################################
        layers.update(self._main_block([352,192,320,192,224,224,128], 'inception_5a', layers['inception_4e_output'],
                        dataDict, poolType = 'AVG', weight_decay=weight_decay))

        ################################### inception_5b  ######################################
        layers.update(self._main_block([352,192,320,192,224,224,128], 'inception_5b', layers['inception_5a_output'],
                        dataDict, poolType = 'MAX', weight_decay=weight_decay))


        ################################### global pool  ######################################

        layers['global_pool'] = avg_pool_layer(input_tensor=layers['inception_5b_output'], filter_dims=[7,7],
                            stride_dims=[1,1], name='global_pool', padding='VALID')


        layers['dropout'] = tf.nn.dropout(layers['global_pool'], keep_prob)




        ################################### loss accuracy  ######################################




        layers['logits'] = fully_connected_layer(input_tensor=layers['dropout'],
                                  out_dim=outputDims,
                                  name='fc',
                                  weight_decay=weight_decay,
                                  non_linear_fn=None,
                                  weight_init=tf.constant_initializer(dataDict['fc']['fc-action_0'].T),
                                  bias_init=tf.constant_initializer(dataDict['fc']['fc-action_1']))

        #layers['logits'] = tf.reshape(layers['fc'], [-1, 3, 1, 51])

    #    layers['logits'] = avg_pool_layer(input_tensor=layers['fc_reshape'], filter_dims=[3,1],
    #                        stride_dims=[1,1], name='segment_consensus', padding='VALID')


        return layers[return_layer]




    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining):
        """
        Args:
            :index:       Integer indicating the index of video frame from the text file containing video lists
            :data:        Data loaded from HDF5 files
            :labels:      Labels for loaded data
            :size:        List detailing values of height and width for final frames
            :is_training: Boolean value indication phase (TRAIN OR TEST)
        """
        return preprocess_tfrecords(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining)



    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        crossEntropyLoss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return crossEntropyLoss



def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]


if __name__=="__main__":

    import os
    with tf.name_scope("my_scope") as scope:
        #inputDims = 32

        dataset = 'UCF101'
        x_placeholder = tf.placeholder(tf.float32, shape=(10,224,224,3))
    #    y = tf.placeholder(tf.int32, [None])

        model = TSN()

        if dataset =='UCF101':
            outDims = 101
        else:
            outDims = 51
        logits = model.inference(x_placeholder, False, 10, outDims, None, scope, dataDict = dataset)


        # Session setup
        sess  = tf.Session()

        # Initialize Variables
        init = tf.global_variables_initializer()
        sess.run(init)


        #softmax  = tf.nn.softmax(logits)
        #fName = 'trainlist'
        fName = 'testlist'

        if fName == 'testlist':
            if dataset == 'UCF101':
                totalVids = 3783
            else:
                totalVids = 1530

        else:
            if dataset=='UCF101':
                totalVids = 9537
            else:
                totalVids = 3570


        vidList = np.arange(totalVids)
        np.random.shuffle(vidList)
        numVids = 0
        acc=0
        for vid in vidList:

            input_data, label = load_data(model, 720, fName, '/z/home/madantrg/Datasets/HMDB51HDF5RGB/Split1',os.path.join('/z/home/erichof/TF_Activity_Recognition_Framework/datasets/HMDB51',fName+'01.txt'), '/z/home/erichof/TF_Activity_Recognition_Framework/datasets/HMDB51/classInd.txt', [224,224], False, 'HMDB51')

            frame_scores = []
            for frame in input_data:
        #        import pdb; pdb.set_trace()
    #            layers = all_layers['conv1'].eval(session=sess, feed_dict={x_placeholder: frame})
                import pdb; pdb.set_trace()
                prediction = all_layers['logits'].eval(session=sess, feed_dict={x_placeholder: frame.astype('int').astype('float32')})
            #    print label
    #            guess = np.mean(prediction, axis=0).argmax()
            #    print "guess: ", guess
            #    import pdb; pdb.set_trace()
                frame_scores.append(prediction)

            mean = softmax(np.mean(frame_scores, axis = 1).mean(axis=0))
            video_pred = np.argmax(mean)
            numVids+=1
            if int(video_pred) == int(label):
                acc+=1
        #    import pdb; pdb.set_trace()
            print 'step, video pred, label, acc: ', numVids, video_pred, label, float(acc)/numVids
        #    print "label: ", label
