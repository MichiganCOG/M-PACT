import os

import numpy      as np
import tensorflow as tf
from tensorflow.python.training import queue_runner
from random import shuffle



def load_dataset(model, num_gpus, batch_size, output_dims, input_dims, seq_length, size, base_data_path, dataset, istraining, clip_length, clip_offset, num_clips, clip_overlap, verbose=True):
    """
    Function load dataset, setup queue and read data into queue
    Args:
        :model:              tf-activity-recognition framework model object
        :num_gpus:           Number of gpus to use when training
        :batch_size:         Number of clips to load into the model each step.
        :input_dims:         Number of frames used in input
        :output_dims:        Integer number of classes in current dataset
        :seq_length:         Length of output sequence expected from LSTM
        :size:               List detailing height and width of frame
        :dataset:            Name of dataset being processed
        :base_data_path:     Full path to root directory containing datasets
        :istraining:         Boolean variable indicating training/testing phase
        :clip_length:        Length of clips to cut video into, -1 indicates using the entire video as one clip')
        :clip_offset:        "none" or "random" indicating where to begin selecting video clips
        :num_clips:          Number of clips to break video into
        :clip_overlap:       Number of frames that overlap between clips, 0 indicates no overlap and -1 indicates clips are randomly selected and not sequential

    Return:
        Input data tensor, label tensor and name of loaded data (video/image)
    """
    # Get a list of tfrecords file names from which to pull videos
    filenames           = []
    number_of_tfrecords = 0

    for f in os.listdir(base_data_path):
        filenames.append(os.path.join(base_data_path,f))
        number_of_tfrecords += 1

    # END FOR

    if verbose:
        print "Number of records available: ", number_of_tfrecords

    # Create Queue which will read in videos num_gpus at a time (Queue seeded for repeatability of experiments)
    tfrecord_file_queue = tf.train.string_input_producer(filenames, shuffle=istraining, name='file_q', seed=0)

    tf.set_random_seed(0) # To ensure the numbers are generated for temporal offset consistently

    if istraining:
        thread_count = 6

    else:
        thread_count = 1

    # END IF

    # Initialize queue that will contain multiple clips of the format [[clip_frame_count, height, width, channels], [labels_copied_seqLength], [name_of_video]]
    clip_q = tf.FIFOQueue(num_gpus*batch_size*thread_count, dtypes=[tf.float32, tf.int32, tf.string], shapes=[[input_dims, size[0], size[1], 3],[seq_length],[]])

    # Attempts to load num_gpus*batch_size number of clips into queue, if there exist too many clips in a video then this function blocks until the clips are dequeued
    enqueue_op = clip_q.enqueue_many(_load_video(model, output_dims, input_dims, seq_length, size, base_data_path, dataset, istraining, clip_length, clip_offset, num_clips, clip_overlap, tfrecord_file_queue))

    # Initialize the queuerunner and add it to the collection, this becomes initialized in train_test_TFRecords_multigpu_model.py after the Session is begun
    qr = tf.train.QueueRunner(clip_q, [enqueue_op]*num_gpus*batch_size*thread_count)
    queue_runner.add_queue_runner(qr)

    # Dequeue the required number of clips so that each gpu contains batch_size clips
    input_data_tensor, labels_tensor, names_tensor = clip_q.dequeue_many(num_gpus*batch_size)

    return input_data_tensor, labels_tensor, names_tensor


def _load_video(model, output_dims, input_dims, seq_length, size, base_data_path, dataset, istraining, clip_length, clip_offset, num_clips, clip_overlap, tfrecord_file_queue):
    """
    Function to load a single video and preprocess its' frames
    Args:
        :model:                tf-activity-recognition framework model object
        :input_dims:           Number of frames used in input
        :output_dims:          Integer number of classes in current dataset
        :seq_length:           Length of output sequence expected from LSTM
        :size:                 List detailing height and width of frame
        :dataset:              Name of dataset being processed
        :base_data_path:       Full path to root directory containing datasets
        :istraining:           Boolean variable indicating training/testing phase
        :clip_length:          Length of clips to cut video into, -1 indicates using the entire video as one clip')
        :clip_offset:          "none" or "random" indicating where to begin selecting video clips
        :num_clips:            Number of clips to break video into
        :clip_overlap:         Number of frames that overlap between clips, 0 indicates no overlap and -1 indicates clips are randomly selected and not sequential
        :tfrecord_file_queue:  A queue containing remaining videos to be loaded for the current epoch

    Return:
        Input data tensor, label tensor and name of loaded data (video/image)
    """

    # Dequeue video data from queue and convert it from TFRecord format (int64 or bytes)
    features = _read_tfrecords(tfrecord_file_queue)
    frames   = tf.cast(features['Frames'], tf.int32)
    height   = tf.cast(features['Height'], tf.int32)
    width    = tf.cast(features['Width'], tf.int32)
    channel  = tf.cast(features['Channels'], tf.int32)
    label    = tf.cast(features['Label'], tf.int32)

    name     = features['Name']

    # Shape [frames, height, width, channels]
    input_data_tensor = tf.reshape(tf.decode_raw(features['Data'], tf.uint8), tf.stack([frames,height,width,channel]))

    # BGR to RGB
    input_data_tensor = input_data_tensor[...,::-1]

    # Reduction in fps to 25 for HMDB51 dataset
    if 'HMDB51' in dataset:
        input_data_tensor, frames, indices = _reduce_fps(input_data_tensor, frames)

    # If clip_length == -1 then the entire video is to be used as a single clip
    if clip_length <= 0:
        clips = [input_data_tensor]
        clips = tf.to_int32(clips)  # Usually occurs within _extract_clips
    else:
        tf.cond(tf.greater(tf.convert_to_tensor(clip_length), frames),lambda:_error_loading_video(),lambda: 1) # Verify that there are not fewer frames than the requested clip_length
        clips = _extract_clips(input_data_tensor, frames, num_clips, clip_offset, clip_length, clip_overlap)

        # tf.Assert(tf.greater(frames, clip_length), [tf.constant("Video ")+name+tf.constant(" contained fewer frames than the specified clip length... Exiting")])


    # clips shape - [num_clips, clip_length or frames, height, width, channels]
    # model.preprocess_tfrecords input shape - [clip_length or frames, height, width, channels]
    # Call preprocessing function related to model chosen that preprocesses each clip as an individual video
    clips_tensor = tf.map_fn(lambda clip:
        model.preprocess_tfrecords(clip, tf.shape(clip)[0], height, width,channel, input_dims, output_dims, seq_length, size, label, istraining),
        clips, dtype=tf.float32)


    num_clips = tf.shape(clips_tensor)[0]
    labels_tensor = tf.tile( [label], [seq_length])
    names_tensor = tf.tile( [name], [num_clips])

    # clips_tensor shape - [num_clips, input_dims, size[0], size[1], channels]
    return [clips_tensor, tf.tile([labels_tensor], [num_clips,1]), names_tensor]


def _read_tfrecords(filename_queue):
    """
    Function that reads and returns the tfrecords of a selected dataset one at a time
    Args:
        :filename_queue:  A queue of all filenames within a dataset

    Return:
        Dictionary containing features of a single sample
    """
    feature_dict = {}
    reader       = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)


    feature_dict['Label']    = tf.FixedLenFeature([], tf.int64)
    feature_dict['Data']     = tf.FixedLenFeature([], tf.string)
    feature_dict['Frames']   = tf.FixedLenFeature([], tf.int64)
    feature_dict['Height']   = tf.FixedLenFeature([], tf.int64)
    feature_dict['Width']    = tf.FixedLenFeature([], tf.int64)
    feature_dict['Channels'] = tf.FixedLenFeature([], tf.int64)
    feature_dict['Name']     = tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example(serialized_example, features=feature_dict)

    return features


def _extract_clips(video, frames, num_clips, clip_offset, clip_length, clip_overlap):
    """
    Function that extracts clips from a video based off of clip specifications
    Args:
        :video:                The video tensor that needs to be split into clips
        :frames:               The number of frames of the video
        :num_clips:            Number of clips to break video into
        :clip_offset:          "none" or "random" indicating where to begin selecting video clips
        :clip_length:          Length of clips to cut video into, -1 indicates using the entire video as one clip')
        :clip_overlap:         Number of frames that overlap between clips, 0 indicates no overlap and -1 indicates clips are randomly selected and not sequential


    Return:
        A tensor containing the clip(s) extracted from the video (shape [clip_number, clip_frames, height, width, channel])
    """
    if clip_offset == 'random':
        video_start = tf.random_uniform([], maxval=frames-1)
    else:
        video_start = 0

    iteration = tf.range(video_start, frames-clip_overlap+1, delta = clip_length-clip_overlap)[:-1]

    if num_clips > 0:
        iteration = iteration[:num_clips]

        # If the clip length caused there to be less clips than the requested num_clips, loop the video for the remaining number of clips
        iteration = tf.cond(tf.greater(num_clips, tf.shape(iteration)[0]),
                            lambda: tf.tile(iteration, [(num_clips-tf.shape(iteration)[0])/num_clips + 1])[:num_clips],
                            lambda: iteration)

    video = tf.to_int32(video)
    clips = tf.map_fn(lambda clip_start: video[clip_start[0][0][0]:clip_start[0][0][0]+clip_length],
                        tf.reshape(iteration, [tf.shape(iteration)[0], 1, 1, 1]))

    return clips


def _reduce_fps(video, frame_count):
    """
    Function that drops frames to match 25 pfs from 30 fps captured videos
    Args:
        :video:       Tensor containing video frames
        :frame_count: Total number of frames in the video

    Return:
        Video with reduced number of frames to match 25fps
    """
    # Convert from 30 fps to 25 fps
    remove_count = tf.cast(tf.ceil(tf.divide(frame_count, 6)), tf.int32)

    intermediate_frames = tf.multiply(remove_count, 5)
    indices = tf.tile([0,1,2,3,4], [remove_count])                                 # [[0,1,2,3,4],[0,1,2,3,4]..]
    indices = tf.reshape(indices, [intermediate_frames])                           # [0,1,2,3,4,0,1,2,3,4,0,1,2....]
    additions = tf.range(remove_count)                                             # [0,1,2,3,4,5,6,....]
    additions = tf.stack([additions, additions, additions, additions, additions])  # [[0,1,2,3,4,5,6...], [0,1,2,3,4,5,6..], [0,1..], [0,1,..], [0,1,...]]
    additions = tf.transpose(additions)                                            # [[0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2], ...]
    additions = tf.reshape(additions, [intermediate_frames])                       # [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3....]
    additions = tf.multiply(additions, 6)                                          # [0,0,0,0,0,6,6,6,6,6,12,12,12,12,12,18,18,18,18,18....]
    indices = tf.add(indices, additions)                                           # [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19....]

    remove_count = tf.cond( tf.equal(frame_count, tf.multiply(remove_count, 6)),
                    lambda: remove_count,
                    lambda: tf.subtract(remove_count, 1))
    output_frames = tf.subtract(frame_count, remove_count)

    indices = tf.slice(indices, [0], [output_frames])
    indices_to_keep = tf.reshape(indices, [output_frames])
    output = tf.gather(video, indices_to_keep)
    return output, output_frames, indices


def _error_loading_video():
    """
    Prints that an error occured while loading the video, indicates that the clip_length was specified to be longer than a videos' frame count
    Args:

    Return:
        returns an integer for tf.cond to function properly in _load_video()
    """
    print "If an error occurs: The video loaded contains fewer frames than the specified clip length."
    return 0
