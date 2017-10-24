# LRCN Model implemnetation for use with tensorflow

import os
import tensorflow as tf
import numpy as np
from rnn_cell_impl import LSTMCell
from lrcn_preprocessing import preprocess

class LRCN():

    def __init__(self, verbose=True):
        self.verbose=verbose
        self.name = 'lrcn'


    def inference(self, inputs, labels, isTraining, inputDims, outputDims, seqLength, scope, dropoutRate = 0.5, return_layer='logits', dataDict=None, cpuId = 0, weight_decay=0.0):

        ############################################################################
        #                        Creating LRCN Network Layers                      #
        ############################################################################


        if self.verbose:
            print('Generating LRCN Network Layers')

        #print('input: ', inputs.shape)

        # Load Model Converted from Caffe
        if dataDict == None:
            dataDict = self.get_dataDict(weight_decay, isTraining)

    #    import pdb;pdb.set_trace()
        with tf.name_scope("my_scope"):#, values=inputs): #scope, 'lrcn', [inputs]):

            layers = {}

            layers['conv1_0'] = tf.nn.bias_add(
                tf.nn.conv2d(input=inputs,
                    filter=dataDict['conv1_0'],
                    strides=[1,2,2,1], padding='VALID', #'VALID'
                    name='conv1'),
                dataDict['conv1_1'])
            layers['conv1'] = tf.nn.relu(
                            layers['conv1_0'],
                            'relu1')

            # layers['1'] = tf.nn.relu(
            #                 tf.nn.bias_add(
            #                     tf.nn.conv2d(input=inputs,
            #                         filter=dataDict['conv1_0'],
            #                         strides=[1,2,2,1], padding='VALID',
            #                         name='conv1'),
            #                     dataDict['conv1_1']),
            #                 'relu1')
            #print('1: ', layers['1'].shape)

            layers['pool1'] = tf.nn.max_pool(value=layers['conv1'], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        #    print('2: ', layers['2'].shape)
            layers['norm1'] = tf.nn.lrn(input=layers['pool1'], depth_radius=2, alpha=0.0001/5, beta=0.75, name='norm1')

        #    print('3: ',layers['3'].shape)


            # # Undo the grouping from Caffe
            # # - Half of the filters get passed over half of the input
            # conv2a = tf.nn.conv2d(input=layers['3'][:,:,:,:48],
            #     filter=dataDict['conv2_0'][:,:,:,:192],
            #     strides=[1,2,2,1], padding='VALID',
            #     name='conv2a')
            #
            #
            # conv2b = tf.nn.conv2d(input=layers['3'][:,:,:,48:],
            #     filter=dataDict['conv2_0'][:,:,:,192:],
            #     strides=[1,2,2,1], padding='VALID',
            #     name='conv2b')
            #
            # print(conv2a.shape)
            #
            # layers['4'] = tf.nn.relu(
            #                 tf.nn.bias_add(
            #                     tf.concat([conv2a, conv2b], 3, name='conv2'),           #however you concatenate that in tensorflow = [256, 5, 5, 48] + [256, 5, 5, 48] = [512, 5, 5, 48]
            #                     dataDict['conv2_1']),
            #                 'relu2')
            # print('4: ',layers['4'].shape)





            from tensorflow.python.ops import array_ops

            input_slices = array_ops.split(layers['norm1'], 2, axis=3)
            kernel_slices = array_ops.split(dataDict['conv2_0'], 2, axis=3)
            output_slices = [tf.nn.conv2d(
                input=input_slice,
                filter=kernel_slice,
                strides=[1,2,2,1],
                padding='VALID')
                for input_slice, kernel_slice in zip(input_slices, kernel_slices)]
            outputs = array_ops.concat(output_slices, axis=3)
            print outputs.shape

            layers['conv2_0'] = tf.nn.bias_add(outputs,dataDict['conv2_1'])
            layers['conv2'] = tf.nn.relu(layers['conv2_0'], 'relu2')
        #    layers['4'] = tf.nn.relu(tf.nn.bias_add(outputs,dataDict['conv2_1']), 'relu2')







            layers['pool2'] = tf.nn.max_pool(value=layers['conv2'], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        #    print('5: ', layers['5'].shape)
            layers['norm2'] = tf.nn.lrn(input=layers['pool2'], depth_radius=2, alpha=0.0001/5, beta=0.75, name='norm2')

        #    print('6: ', layers['6'].shape)

            layers['conv3_0'] = tf.nn.bias_add(
                tf.nn.conv2d(input=layers['norm2'],
                    filter=dataDict['conv3_0'],
                    strides=[1,1,1,1], padding='SAME',
                    name='conv3'),
                dataDict['conv3_1'])

            layers['conv3'] = tf.nn.relu(layers['conv3_0'],'relu3')

        #    print('7: ', layers['7'].shape)

            #
            # # Undo the grouping from Caffe
            # # - Half of the filters get passed over half of the input
            # conv4a = tf.nn.conv2d(input=layers['7'][:,:,:,:256],
            #     filter=dataDict['conv4_0'][:,:,:,:256],
            #     strides=[1,1,1,1], padding='SAME',
            #     name='conv4a')
            #
            #
            # conv4b = tf.nn.conv2d(input=layers['7'][:,:,:,256:],
            #     filter=dataDict['conv4_0'][:,:,:,256:],
            #     strides=[1,1,1,1], padding='SAME',
            #     name='conv4a')
            #
            #
            # layers['8'] = tf.nn.relu(
            #                 tf.nn.bias_add(
            #                     tf.concat([conv4a, conv4b], 3, name='conv4'),           #however you concatenate that in tensorflow = [256, 5, 5, 48] + [256, 5, 5, 48] = [512, 5, 5, 48]
            #                     dataDict['conv4_1']),
            #                 'relu4')
            #
            # print('8: ', layers['8'].shape)
            #
            #
            #
            #



            input_slices = array_ops.split(layers['conv3'], 2, axis=3)
            kernel_slices = array_ops.split(dataDict['conv4_0'], 2, axis=3)
            output_slices = [tf.nn.conv2d(
                input=input_slice,
                filter=kernel_slice,
                strides=[1,1,1,1],
                padding='SAME')
                for input_slice, kernel_slice in zip(input_slices, kernel_slices)]
            outputs = array_ops.concat(output_slices, axis=3)
            #print outputs.shapetf.nn.bias_add(outputs,dataDict['conv4_1'])
            layers['conv4_0'] =tf.nn.bias_add(outputs,dataDict['conv4_1'])
            layers['conv4'] = tf.nn.relu(layers['conv4_0'], 'relu4')




#  https://github.com/tensorflow/tensorflow/pull/10482/files#diff-26aa645fdaefe1f89103555b9c0da70eL433


            # # Undo the grouping from Caffe
            # # - Half of the filters get passed over half of the input
            # conv5a = tf.nn.conv2d(input=layers['8'][:,:,:,:256],
            #     filter=dataDict['conv5_0'][:,:,:,:192],
            #     strides=[1,1,1,1], padding='SAME',
            #     name='conv5a')
            #
            #
            # conv5b = tf.nn.conv2d(input=layers['8'][:,:,:,256:],
            #     filter=dataDict['conv5_0'][:,:,:,192:],
            #     strides=[1,1,1,1], padding='SAME',
            #     name='conv5a')
            #
            #
            # layers['9'] = tf.nn.relu(
            #                 tf.nn.bias_add(
            #                     tf.concat([conv5a, conv5b], 3, name='conv5'),           #however you concatenate that in tensorflow = [256, 5, 5, 48] + [256, 5, 5, 48] = [512, 5, 5, 48]
            #                     dataDict['conv5_1']),
            #                 'relu5')
            #
            # print('9: ', layers['9'].shape)
            #
            #
            #
            #




            input_slices = array_ops.split(layers['conv4'], 2, axis=3)
            kernel_slices = array_ops.split(dataDict['conv5_0'], 2, axis=3)
            output_slices = [tf.nn.conv2d(
                input=input_slice,
                filter=kernel_slice,
                strides=[1,1,1,1],
                padding='SAME')
                for input_slice, kernel_slice in zip(input_slices, kernel_slices)]
            outputs = array_ops.concat(output_slices, axis=3)
    #        print outputs.shape
            layers['conv5'] = tf.nn.relu(tf.nn.bias_add(outputs,dataDict['conv5_1']), 'relu5')









            layers['pool5'] = tf.nn.max_pool(value=layers['conv5'], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')


            batch_size = layers['pool5'].get_shape().as_list()[0]
        #    print(batch_size)
            dense_dim = dataDict['fc6_0'].get_shape().as_list()[0]
    #        print(dense_dim)
        #    print("fc6 ", layers['10'].get_shape())
            fc6 = tf.reshape(layers['pool5'], [batch_size, dense_dim])

            layers['fc6'] = tf.nn.relu(tf.matmul(fc6, dataDict['fc6_0']) + dataDict['fc6_1'], 'relu6')
            print isTraining
            with tf.device('/gpu:'+str(cpuId)):
                if isTraining:
                    layers['fc6'] = tf.reshape(layers['fc6'], shape = [1, 16, 4096])
                else:
                    layers['fc6'] = tf.reshape(layers['fc6'], shape=[10,16,4096])

            #    print('11', layers['11'].shape)
                lstm_cell = LSTMCell(256, forget_bias=0.0, weights_initializer=dataDict['lstm1_0'], bias_initializer=dataDict['lstm1_1'])#, initializer = (dataDict['lstm1_0'], dataDict['lstm1_1']))

                print(lstm_cell.state_size)
                layers['rnn_outputs'], states = tf.nn.dynamic_rnn(lstm_cell, layers['fc6'], dtype=tf.float32)#, initial_state=dataDict['lstm1_2'])#, dtype=tf.float32)#[tf.reshape(layers['11'], [16,10,4096])], dtype=tf.float32)#, initial_state=dataDict['lstm1_2'])
            layers['rnn_outputs_rs'] = tf.reshape(layers['rnn_outputs'], shape=[-1,256])
        #    print('12: ', layers['12'])
        #    print('states: ', states)
            layers['logits'] = tf.matmul(layers['rnn_outputs_rs'], dataDict['fc8_0']) + dataDict['fc8_1']

        #    print('output: ', layers['output'])
            return layers[return_layer]






    def get_dataDict(self, decay, isTraining):

        # conv1_0 = np.load('Numpy_from_Caffe/conv1_0.npy')
        # conv1_1 = np.load('Numpy_from_Caffe/conv1_1.npy')
        # conv2_0 = np.load('Numpy_from_Caffe/conv2_0.npy')
        # conv2_1 = np.load('Numpy_from_Caffe/conv2_1.npy')
        # conv3_0 = np.load('Numpy_from_Caffe/conv3_0.npy')
        # conv3_1 = np.load('Numpy_from_Caffe/conv3_1.npy')
        # conv4_0 = np.load('Numpy_from_Caffe/conv4_0.npy')
        # conv4_1 = np.load('Numpy_from_Caffe/conv4_1.npy')
        # conv5_0 = np.load('Numpy_from_Caffe/conv5_0.npy')
        # conv5_1 = np.load('Numpy_from_Caffe/conv5_1.npy')
        # fc6_0 = np.load('Numpy_from_Caffe/fc6_0.npy')
        # fc6_1 = np.load('Numpy_from_Caffe/fc6_1.npy')
        # lstm1_0 = np.load('Numpy_from_Caffe/lstm1_0.npy')
        # lstm1_1 = np.load('Numpy_from_Caffe/lstm1_1.npy')
        # lstm1_2 = np.load('Numpy_from_Caffe/lstm1_2.npy')
        # fc8_final_0 = np.load('Numpy_from_Caffe/fc8-final_0.npy')
        # fc8_final_1 = np.load('Numpy_from_Caffe/fc8-final_1.npy')
        conv1_0 = np.load('models/lrcn/Numpy_from_Caffe/conv1_0.npy')
    #    conv1_0[0][0][:][:] = conv1_0[0][0][:][:].T
    #    conv1_0[0][1][:][:] = conv1_0[0][1][:][:].T
    #    conv1_0[0][2][:][:] = conv1_0[0][2][:][:].T
        conv1_1 = np.load('models/lrcn/Numpy_from_Caffe/conv1_1.npy')
        conv2_0 = np.load('models/lrcn/Numpy_from_Caffe/conv2_0.npy')
        conv2_1 = np.load('models/lrcn/Numpy_from_Caffe/conv2_1.npy')
        conv3_0 = np.load('models/lrcn/Numpy_from_Caffe/conv3_0.npy')
        conv3_1 = np.load('models/lrcn/Numpy_from_Caffe/conv3_1.npy')
        conv4_0 = np.load('models/lrcn/Numpy_from_Caffe/conv4_0.npy')
        conv4_1 = np.load('models/lrcn/Numpy_from_Caffe/conv4_1.npy')
        conv5_0 = np.load('models/lrcn/Numpy_from_Caffe/conv5_0.npy')
        conv5_1 = np.load('models/lrcn/Numpy_from_Caffe/conv5_1.npy')
        fc6_0 = np.load('models/lrcn/Numpy_from_Caffe/fc6_0.npy')
        fc6_0 = fc6_0.reshape((4096, 384, 6, 6))
        fc6_0 = fc6_0.transpose((2,3,1,0))
        fc6_0 = fc6_0.reshape(384*6*6, 4096)
        fc6_1 = np.load('models/lrcn/Numpy_from_Caffe/fc6_1.npy')
        lstm1_0 = np.load('models/lrcn/Numpy_from_Caffe/lstm1_0.npy')
        lstm1_1 = np.load('models/lrcn/Numpy_from_Caffe/lstm1_1.npy')
        lstm1_2 = np.load('models/lrcn/Numpy_from_Caffe/lstm1_2.npy')
        fc8_final_0 = np.load('models/lrcn/Numpy_from_Caffe/fc8-final_0.npy')
        fc8_final_1 = np.load('models/lrcn/Numpy_from_Caffe/fc8-final_1.npy')

        print convert_NCHW_to_NHWC(conv1_0).shape

        with tf.name_scope("my_scope"):
            dataDict = {}
            dataDict['conv1_0'] = tf.get_variable('conv1_0', [7,7,3,96], initializer=tf.constant_initializer(convert_NCHW_to_filter_shape(conv1_0)))
            dataDict['conv1_1'] = tf.get_variable('conv1_1', [96], initializer=tf.constant_initializer(conv1_1))
            dataDict['conv2_0'] = tf.get_variable('conv2_0', [5,5,48,384], initializer=tf.constant_initializer(convert_NCHW_to_filter_shape(conv2_0)))
            dataDict['conv2_1'] = tf.get_variable('conv2_1', [384], initializer=tf.constant_initializer(conv2_1))
            dataDict['conv3_0'] = tf.get_variable('conv3_0', [3, 3, 384, 512], initializer=tf.constant_initializer(convert_NCHW_to_filter_shape(conv3_0)))
            dataDict['conv3_1'] = tf.get_variable('conv3_1', [512], initializer=tf.constant_initializer(conv3_1))
            dataDict['conv4_0'] = tf.get_variable('conv4_0', [3, 3 , 256,512], initializer=tf.constant_initializer(convert_NCHW_to_filter_shape(conv4_0)))
            dataDict['conv4_1'] = tf.get_variable('conv4_1', [512], initializer=tf.constant_initializer(conv4_1))
            dataDict['conv5_0'] = tf.get_variable('conv5_0', [3, 3, 256, 384], initializer=tf.constant_initializer(convert_NCHW_to_filter_shape(conv5_0)))
            dataDict['conv5_1'] = tf.get_variable('conv5_1', [384], initializer=tf.constant_initializer(conv5_1))

            dataDict['fc6_0'] = tf.get_variable('fc6_0', [13824, 4096], initializer=tf.constant_initializer(fc6_0))#np.transpose(fc6_0, [1,0])))
            dataDict['fc6_1'] = tf.get_variable('fc6_1', [4096], initializer=tf.constant_initializer(fc6_1))

            dataDict['lstm1_0'] = tf.get_variable('rnn/lstm_cell/kernel', [4352, 1024], initializer=tf.constant_initializer(convert_lstm_weights(lstm1_0, lstm1_2)))
            dataDict['lstm1_1'] = tf.get_variable('rnn/lstm_cell/bias', [1024], initializer=tf.constant_initializer(lstm1_1))

            dataDict['fc8_0'] = tf.get_variable('fc8_0', [256, 101], initializer=tf.constant_initializer(np.transpose(fc8_final_0, [1,0])))
            dataDict['fc8_1'] = tf.get_variable('fc8_1', [101], initializer=tf.constant_initializer(fc8_final_1))
            if isTraining:
                for layer in dataDict:
                    print "layer", dataDict[layer]
                    weight_decay(dataDict[layer], decay)


        return dataDict



    # For weight decay
    # def loss(self, logits, labels):
    #     cross_entropy_mean = tf.nn.spare_softmax_crooss_entropy_with_logits(logit, labels)
    #     tf.add_to_collection('losses', cross_entropy_mean)
    #     loss = tf.add_n(tf.get_collection('losess'), name='total_loss')
    #     return loss


    def preprocess(self, index, data, labels, size, isTraining):
        return preprocess(index, data,labels, size, isTraining)


    """ Function to return loss calculated on given network """
    def loss(self, logits, labels):

        labels = tf.cast(labels, tf.int64)
        crossEntropyLoss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                        logits=logits)

        #crossEntropyMean = tf.reduce_mean(crossEntropyLoss, name='crossEntropy')

        #tf.add_to_collection('losses', crossEntropyMean)
        #return tf.add_n(tf.get_collection('losses'), name='total_loss')
        return crossEntropyLoss

def convert_NCHW_to_NHWC(layer):        # move to whatever generate dataDict, needs to be done before tf.get_variable() is called, the weights get moved into here
    return tf.transpose(layer, [0,2,3,1])    #NCHW to NHWC


def convert_NCHW_to_filter_shape(layer):
    return np.transpose(layer, [2,3,1,0])   # NCHW to [filter_height, filter_width, in_channels, out_channels]  described in tf.nn.conv2d for filters https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

def convert_lstm_weights(input_weights, hidden_layer_weights):
    layer = np.concatenate((input_weights, hidden_layer_weights), axis=1)
    return np.transpose(layer)



def weight_decay(var, decay):
    if decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)










if __name__ == '__main__':




#    inputs = np.zeros((160,227,227,3), dtype=np.float32)
    input_placeholder = tf.placeholder(tf.float32, shape=(160, 227,227,3))

    lrcn = LRCN()
    layers = lrcn.inference(input_placeholder, 0, False, 1, decay=0.001)
    logits = tf.nn.softmax(layers)
    print logits
#    sess = tf.Session()
#    init  =tf.global_variables_initializer()
#    sess.run(init)
#    print logits.eval(session=sess, feed_dict={input_placeholder: inputs})
