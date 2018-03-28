import argparse
import tensorflow as tf
import numpy as np
import os
from utils import make_dir
import cv2

# Definition of arguments used in functions defined within this file

parser = argparse.ArgumentParser()

parser.add_argument('--videos_dir', action='store',
        help = 'Directory containing directories of acions with videos therein')
parser.add_argument('--save_dir', action='store',
        help = 'Directory to save tfrecords files to')

args = parser.parse_args()


'''

Assumes file structure of action_class/video_name.ext
All action_class folders in the one directory

NOTE: First manually separate training, testing, and validation lists
'''


def _int64(value):
    """
    Cast a value to int64 list
    Args:
        :value: Value to be casted to int64 list

    Returns:
        Int64 converted value list 
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes(value):
    """
    Cast a value to byte list
    Args:
        :value: Value to be casted to byte list

    Returns:
        Byte converted value list 
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_tfrecords(data, label, vidname, save_dir):
    """
    Save given data as tfrecords 
    Args:
        :data:     Data to be saved as tfrecord
        :label:    Corresponding labels of Data to be saved in tfrecord 
        :vidname:  Name of file to be saved as tfrecord
        :save_dir: Directory where tfrecord needs to be saved

    Returns:
        Nothing 
    """

    filename = os.path.join(save_dir, vidname+'.tfrecords')
    writer   = tf.python_io.TFRecordWriter(filename)

    features             = {}
    features['Label']    = _int64(label)
    features['Data']     = _bytes(np.array(data).tostring())
    features['Frames']   = _int64(data.shape[0])
    features['Height']   = _int64(data.shape[1])
    features['Width']    = _int64(data.shape[2])
    features['Channels'] = _int64(data.shape[3])
    features['Name']     = _bytes(str(vidname))


    example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(example.SerializeToString())
    writer.close()



def load_video_data_from_file(video_path):
    """
    Load video data from a specified file 
    Args:
        :video_path: Full path from which to read video data

    Returns:
        Data read from video as numpy array 
    """

    video       = cv2.VideoCapture(video_path)
    flag, frame = video.read()

    count = 0
    data1 = np.array([])
    data2 = np.array([])

    while flag:
        H,W,C = frame.shape

        if count < 150:
            if count == 0:
                data1 = frame.reshape(1,H,W,C)

            else:
                data1 = np.concatenate((data1, frame.reshape(1,H,W,C)))

            # END IF

        else:
            if count == 150:
                data2 = frame.reshape(1,H,W,C)

            else:
                data2 = np.concatenate((data2, frame.reshape(1,H,W,C)))

            # END IF

        # END IF

        count += 1
        flag, frame = video.read()

    if len(data2)!=0:
        data = np.concatenate((data1, data2))

    else:
        data = np.array(data1)

    # END IF

    return data


def convert_dataset(videos_dir, save_dir):
    """
    Function to convert any given dataset to tfrecords 
    Args:
        :videos_dir: Full path to directory containing action specific folders
        :save_dir:   Full path to directory in which tfrecords need to be saved 

    Returns:
        Nothing 
    """

    actions = os.listdir(videos_dir)
    actions = np.array(map(lambda x: x.lower(), actions))
    actions.sort()
    actions = actions.tolist()

    for action in actions:
        for video in os.listdir(os.path.join(videos_dir, action)):
            data = load_video_data_from_file(os.path.join(videos_dir, action, video))
            save_tfrecords(data, actions.index(action), action+'_'+video, save_dir)

        # END FOR

    # END FOR




if __name__=='__main__':

    print "Provide as single directory of a dataset splits to convert to tfrecords (--videos_dir). Directory must include subdirectories of action classes in the dataset. Each subdirectory includes all video files to be converted fo that action class."
    print "First ensure that training, testing, and validation dataset splits have been separated."
    print "Also provide a single directory to save all tfrecords files to (--save_dir)."


    convert_dataset(args.videos_dir, args.save_dir)
