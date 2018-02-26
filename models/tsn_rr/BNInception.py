from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def BNInception(inputs,
                num_classes=51,
                is_training=True,
                is_init=False,
                dropout_rate=0.8,
                weight_decay=0.0005,
                scope=None):

    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=None, biases_initializer=tf.constant_initializer(0.2),
                                       weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.batch_norm], trainable=False, is_training=False, activation_fn=tf.nn.relu, scale=True, epsilon=1e-5):
            # When finetuning -> is_training of the first Bn should also be False
            layers = slim.conv2d(inputs, 64, [7,7], stride=2, scope='conv1/7x7_s2')
            if is_init:
                layers = slim.batch_norm(layers, trainable=is_training, is_training=is_training, scope='conv1/7x7_s2_bn')
            else:
                layers = slim.batch_norm(layers, scope='conv1/7x7_s2_bn')
            layers = slim.max_pool2d(layers, [3,3], 2, scope='pool1/3x3_s2')

            layers = slim.conv2d(layers, 64, [1,1], scope='conv2/3x3_reduce')
            layers = slim.batch_norm(layers, scope='conv2/3x3_reduce_bn')
            layers = slim.conv2d(layers, 192, [3,3], scope='conv2/3x3')
            layers = slim.batch_norm(layers, scope='conv2/3x3_bn')
            layers = slim.max_pool2d(layers, [3,3], 2, scope='pool2/3x3_s2')

            layers = _inception_block_1(layers, [64, 64, 64, 64, 96, 96, 32], scope='inception_3a')
            layers = _inception_block_1(layers, [64, 64, 96, 64, 96, 96, 64], scope='inception_3b')
            layers = _inception_block_2(layers, [128, 160, 64, 96, 96], scope='inception_3c')
            layers = _inception_block_1(layers, [224, 64, 96, 96, 128, 128, 128], scope='inception_4a')
            layers = _inception_block_1(layers, [192, 96, 128, 96, 128, 128, 128], scope='inception_4b')
            layers = _inception_block_1(layers, [160, 128, 160, 128, 160, 160, 128], scope='inception_4c')
            layers = _inception_block_1(layers, [96, 128, 192, 160, 192, 192, 128], scope='inception_4d')
            layers = _inception_block_2(layers, [128, 192, 192, 256, 256], scope='inception_4e')
            layers = _inception_block_1(layers, [352, 192, 320, 160, 224, 224, 128], scope='inception_5a')
            layers = _inception_block_1(layers, [352, 192, 320, 192, 224, 224, 128], scope='inception_5b', pool='max')

            layers = slim.avg_pool2d(layers, [7,7], 1, scope='global_pool')
            #layers = tf.reduce_mean(layers, [1,2], name='global_pool')
            layers = slim.dropout(layers, dropout_rate, scope='dropout')
            #layers = slim.flatten(layers)
            layers = slim.fully_connected(layers, num_classes, activation_fn=None, weights_initializer=trunc_normal(0.001), weights_regularizer=slim.l2_regularizer(weight_decay), scope='fc-action')

    return layers

def _inception_block_1(layers, output_list, scope='', pool='avg'):
    assert len(output_list)==7, 'not the first type of inception block'
    branch_1 = slim.conv2d(layers, output_list[0], [1,1], scope=scope+'/1x1')
    branch_1 = slim.batch_norm(branch_1, scope=scope+'/1x1_bn')
    branch_2 = slim.conv2d(layers, output_list[1], [1,1], scope=scope+'/3x3_reduce')
    branch_2 = slim.batch_norm(branch_2, scope=scope+'/3x3_reduce_bn')
    branch_2 = slim.conv2d(branch_2, output_list[2], [3,3], scope=scope+'/3x3')
    branch_2 = slim.batch_norm(branch_2, scope=scope+'/3x3_bn')
    branch_3 = slim.conv2d(layers, output_list[3], [1,1], scope=scope+'/double_3x3_reduce')
    branch_3 = slim.batch_norm(branch_3, scope=scope+'/double_3x3_reduce_bn')
    branch_3 = slim.conv2d(branch_3, output_list[4], [3,3], scope=scope+'/double_3x3_1')
    branch_3 = slim.batch_norm(branch_3, scope=scope+'/double_3x3_1_bn')
    branch_3 = slim.conv2d(branch_3, output_list[5], [3,3], scope=scope+'/double_3x3_2')
    branch_3 = slim.batch_norm(branch_3, scope=scope+'/double_3x3_2_bn')
    if pool=='max':
        branch_4 = slim.max_pool2d(layers, [3,3], 1, padding='SAME', scope=scope+'/pool')
    elif pool=='avg':
        branch_4 = slim.avg_pool2d(layers, [3,3], 1, padding='SAME', scope=scope+'/pool')
    branch_4 = slim.conv2d(branch_4, output_list[6], [1,1], scope=scope+'/pool_proj')
    branch_4 = slim.batch_norm(branch_4, scope=scope+'/pool_proj_bn')
    block = tf.concat([branch_1, branch_2, branch_3, branch_4], 3, name=scope+'/output')
    #block = tf.concat([branch_4, branch_3, branch_2, branch_1], 3, name=scope+'/output')
    return block

def _inception_block_2(layers, output_list, scope=''):
    assert len(output_list)==5, 'not the second type of inception block'
    branch_1 = slim.conv2d(layers, output_list[0], [1,1], scope=scope+'/3x3_reduce')
    branch_1 = slim.batch_norm(branch_1, scope=scope+'/3x3_reduce_bn')
    branch_1 = slim.conv2d(branch_1, output_list[1], [3,3], stride=2, scope=scope+'/3x3')
    branch_1 = slim.batch_norm(branch_1, scope=scope+'/3x3_bn')
    branch_2 = slim.conv2d(layers, output_list[2], [1,1], scope=scope+'/double_3x3_reduce')
    branch_2 = slim.batch_norm(branch_2, scope=scope+'/double_3x3_reduce_bn')
    branch_2 = slim.conv2d(branch_2, output_list[3], [3,3], scope=scope+'/double_3x3_1')
    branch_2 = slim.batch_norm(branch_2, scope=scope+'/double_3x3_1_bn')
    branch_2 = slim.conv2d(branch_2, output_list[4], [3,3], stride=2, scope=scope+'/double_3x3_2')
    branch_2 = slim.batch_norm(branch_2, scope=scope+'/double_3x3_2_bn')
    branch_3 = slim.max_pool2d(layers, [3,3], 2, padding='SAME', scope=scope+'/pool') #check the padding
    block = tf.concat([branch_1, branch_2, branch_3], 3, name=scope+'/output')
    #block = tf.concat([branch_3, branch_2, branch_1], 3, name=scope+'/output')
    return block
