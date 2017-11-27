import os

import numpy      as np
import tensorflow as tf

from random import shuffle


def load_dataset(model, num_gpus, output_dims, input_dims, seq_length, size, base_data_path, dataset, istraining):

    # Get a list of tfrecords file names to pull videos from
    filenames = []
    number_of_tfrecords = 0
    for f in os.listdir(base_data_path):
        filenames.append(os.path.join(base_data_path,f))
        number_of_tfrecords += 1
    print "Number of records available: ", number_of_tfrecords

    # Create Queue which will read in videos num_gpus at a time
    tfrecord_file_queue = tf.train.string_input_producer(filenames, shuffle=istraining, name='q', seed=0)

    tf.set_random_seed(0) # To ensure the numbers are generated for temporal offset consistently

    input_data_list     = []
    labels_list 	= []
    names_list 		= []

    # Read in num_gpus number of videos from queue
    for gpu_idx in range(num_gpus):

        # Dequeue video data from queue and convert it from TFRecord format (int64 or bytes)
        features = _read_tfrecords(tfrecord_file_queue)
        frames = tf.cast(features['Frames'], tf.int32)
        height = tf.cast(features['Height'], tf.int32)
        width = tf.cast(features['Width'], tf.int32)
        channel = tf.cast(features['Channels'], tf.int32)
        name = features['Name']
        label = tf.cast(features['Label'], tf.int32)
        input_data_tensor = tf.reshape(tf.decode_raw(features['Data'], tf.uint8), tf.stack([frames,height,width,channel]))

        if 'HMDB51' in dataset:
            input_data_tensor, frames, indices = _reduce_fps(input_data_tensor, frames)

	input_data_tensor, labels_tensor = model.preprocess_tfrecords(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining)
	
        input_data_list.append(input_data_tensor)
        labels_list.append(labels_tensor)
        names_list.append(name)

    input_data_tensor = tf.convert_to_tensor(input_data_list)
    labels_tensor = tf.convert_to_tensor(labels_list)
    names         = tf.convert_to_tensor(names_list)
    return input_data_tensor, labels_tensor, names


def _read_tfrecords(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature_dict = {}
    feature_dict['Label'] = tf.FixedLenFeature([], tf.int64)
    feature_dict['Data'] = tf.FixedLenFeature([], tf.string)
    feature_dict['Frames'] =  tf.FixedLenFeature([], tf.int64)
    feature_dict['Height'] =  tf.FixedLenFeature([], tf.int64)
    feature_dict['Width'] =  tf.FixedLenFeature([], tf.int64)
    feature_dict['Channels'] =  tf.FixedLenFeature([], tf.int64)
    feature_dict['Name'] = tf.FixedLenFeature([], tf.string)
    features = tf.parse_single_example(serialized_example, features=feature_dict)
    return features

def _reduce_fps(video, frame_count):
    # Convert from 30 fps to 25 fps
    remove_count = tf.cast(tf.ceil(tf.divide(frame_count, 6)), tf.int32)
    output_frames = tf.subtract(frame_count, remove_count)
    intermediate_frames = tf.multiply(remove_count, 5)
    indices = tf.tile([1,2,3,4,5], [remove_count])                                 # [[1,2,3,4,5],[1,2,3,4,5]..]
    indices = tf.reshape(indices, [intermediate_frames])                           # [1,2,3,4,5,1,2,3,4,5,1,2....]
    additions = tf.range(remove_count)                                             # [0,1,2,3,4,5,6,....]
    additions = tf.stack([additions, additions, additions, additions, additions])  # [[0,1,2,3,4,5,6...], [0,1,2,3,4,5,6..], [0,1..], [0,1,..], [0,1,...]]
    additions = tf.transpose(additions)                                            # [[0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2], ...]
    additions = tf.reshape(additions, [intermediate_frames])                       # [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3....]
    additions = tf.multiply(additions, 6)                                          # [0,0,0,0,0,6,6,6,6,6,12,12,12,12,12,18,18,18,18,18....]
    indices = tf.add(indices, additions)                                           # [1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19....]
    indices = tf.slice(indices, [0], [output_frames])
    indices_to_keep = tf.reshape(indices, [output_frames])
    output = tf.gather(video, indices_to_keep)
    return output, output_frames, indices
