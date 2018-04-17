import tensorflow as tf
import numpy as np
from utils.preprocessing_utils import *


def preprocess_for_train(image, output_height, output_width):
    """Preprocesses the given image for training.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    Returns:
    A preprocessed image.
    """
    #image = random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    #image = tf.image.random_flip_left_right(image)
    return image


def preprocess_for_eval(image, output_height, output_width):
    """Preprocesses the given image for evaluation.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    Returns:
    A preprocessed image.
    """
    image = central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return image


def preprocess_image(image, output_height, output_width, is_training=False):
    """Preprocesses the given image.
    Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    Returns:
    A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, output_height, output_width)
    else:
        return preprocess_for_eval(image, output_height, output_width)

    # END IF



def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, input_alpha=1.0):
    """
    Preprocessing function corresponding to the chosen model
    Args:
        :input_data_tensor: Raw input data
        :frames:            Total number of frames
        :height:            Height of frame
        :width:             Width of frame
        :channel:           Total number of color channels
        :input_dims:        Number of frames to be provided as input to model
        :output_dims:       Total number of labels
        :seq_length:        Number of frames expected as output of model
        :size:              Output size of preprocessed frames
        :label:             Label of current sample
        :istraining:        Boolean indicating training or testing phase

    Return:
        Preprocessing input data and labels tensor
    """

    _mean_image = np.load('models/weights/sport1m_train16_128_mean.npy')[0]
    _mean_image = _mean_image.transpose(1,2,3,0)


    # Convert to BGR as used by the original authors
    input_data_tensor = input_data_tensor[...,::-1]

    num_frames_per_clip = input_dims

    input_data_tensor = tf.cast(input_data_tensor, tf.float32)

    input_data_tensor = resample_input(input_data_tensor, frames, frames, input_alpha)

    input_data_tensor = tf.map_fn(lambda img: resize(img, 128, 171), input_data_tensor)

    input_data_tensor = input_data_tensor - _mean_image[...,::-1].tolist()

    if istraining:
        input_data_tensor = random_crop_clip(input_data_tensor, size[0], size[1])
        input_data_tensor = random_flip_left_right_clip(input_data_tensor)

    input_data_tensor = tf.map_fn(lambda img: preprocess_image(img, size[0], size[1], is_training=istraining), input_data_tensor)


    return input_data_tensor
