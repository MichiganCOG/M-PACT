import tensorflow as tf
import numpy      as np

from utils.preprocessing_utils import *


def preprocess_for_train(image, output_height, output_width, resize_side):
  """Preprocesses the given image for training.
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.
  Returns:
    A preprocessed image.
  """

  ############################################################################
  #             TODO: Add preprocessing done during training phase           #
  #         Preprocessing option found in utils/preprocessing_utils.py       #
  #                                                                          #
  #  EX:    image = aspect_preserving_resize(image, resize_side)             #
  #         image = central_crop([image], output_height, output_width)[0]    #
  #         image.set_shape([output_height, output_width, 3])                #
  #         image = tf.to_float(image)                                       #
  #         return image                                                     #
  ############################################################################


def preprocess_for_eval(image, output_height, output_width, resize_side):
  """Preprocesses the given image for evaluation.
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.
  Returns:
    A preprocessed image.
  """

  ############################################################################
  #             TODO: Add preprocessing done during training phase           #
  #         Preprocessing option found in utils/preprocessing_utils.py       #
  #                                                                          #
  #  EX:    image = aspect_preserving_resize(image, resize_side)             #
  #         image = central_crop([image], output_height, output_width)[0]    #
  #         image.set_shape([output_height, output_width, 3])                #
  #         image = tf.to_float(image)                                       #
  #         return image                                                     #
  ############################################################################


def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=RESIZE_SIDE_MIN):
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
      return preprocess_for_train(image, output_height, output_width,
                             resize_side_min)

  else:
      return preprocess_for_eval(image, output_height, output_width,
                             resize_side_min)

  # END IF



def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step, input_alpha=1.0):
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

    # Allow for resampling of input during testing for evaluation of the model's stability over video speeds
    input_data_tensor = tf.cast(input_data_tensor, tf.float32)
    input_data_tensor = resample_input(input_data_tensor, frames, frames, input_alpha)

    # Apply preprocessing related to individual frames (cropping, flipping, resize, etc.... )
    input_data_tensor = tf.map_fn(lambda img: preprocess_image(img, size[0], size[1], is_training=istraining, resize_side_min=size[0]), input_data_tensor)


    ##########################################################################################################################
    #                                                                                                                        #
    # TODO: Add any video related preprocessing (looping, resampling, etc.... Options found in utils/preprocessing_utils.py) #
    #                                                                                                                        #
    ##########################################################################################################################


    return input_data_tensor
