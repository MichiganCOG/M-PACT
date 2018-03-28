import tensorflow as tf
import numpy as np
from utils.preprocessing_utils import *

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512

def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=256,
                         resize_side_max=512):
  """Preprocesses the given image for training.
  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.
  Returns:
    A preprocessed image.
  """
  image = aspect_preserving_resize(image, resize_side_min)
  image = central_crop([image], output_height, output_width)[0]

  image.set_shape([output_height, output_width, 3])

  image = tf.to_float(image)
  image = image * 2./255. - 1.

  return image

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
  image = aspect_preserving_resize(image, resize_side)
  image = central_crop([image], output_height, output_width)[0]

  image.set_shape([output_height, output_width, 3])

  image = tf.to_float(image)
  image = image * 2./255. - 1.

  return image


def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=256,
                     resize_side_max=512):
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
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].
  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, output_height, output_width,
                               resize_side_min, resize_side_max)

  else:
    return preprocess_for_eval(image, output_height, output_width,
                               resize_side_min)

  # END IF

def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, input_alpha, tracker):
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
        :input_alpha:       Desired resampling rate (input)
        :tracker:           Number of complete videos loaded

    Return:
        Preprocessing input data and labels tensor
    """
    # Setup different temporal footprints for training and testing phase
    if istraining:
        footprint = 64

    else:
        footprint = 250

    # END IF

    # Ensure that sufficient frames exist in input to extract 250 frames (assuming a 5 sec temporal footprint)
    temporal_offset   = tf.cond(tf.greater(frames, 250), lambda: tf.random_uniform(dtype=tf.int32, minval=0, maxval=frames - 250 + 1, shape=np.asarray([1]))[0], lambda: tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=np.asarray([1]))[0])

    input_data_tensor = tf.cond(tf.less(frames - temporal_offset, 250),
                                lambda: loop_video_with_offset(input_data_tensor[temporal_offset:,:,:,:], input_data_tensor, frames-temporal_offset, frames, height, width, channel, 250),
                                lambda: input_data_tensor[temporal_offset:temporal_offset + 250, :, :, :])

    # Remove excess frames after looping to reduce to footprint size
    input_data_tensor = tf.slice(input_data_tensor, [0,0,0,0], tf.stack([250, height, width, channel]))
    input_data_tensor = tf.reshape(input_data_tensor, tf.stack([250, height, width, channel]))
    input_data_tensor = tf.cast(input_data_tensor, tf.float32)

    # Resample input to desired rate (input fluctuation only, not related to model)
    input_data_tensor = resample_input(input_data_tensor, 250, 250, input_alpha)

    # Resample input to desired rate (resampling as a model requirement)
    if istraining:
        input_data_tensor, alpha_tensor = resample_model_sinusoidal(input_data_tensor, footprint, 250, tracker)

    else:
        input_data_tensor = resample_model(input_data_tensor, footprint, 250, 1.0)
        alpha_tensor = tf.convert_to_tensor(1.0)

    # END IF

    # Preprocess data
    input_data_tensor = tf.map_fn(lambda img: preprocess_image(img, size[0], size[1], is_training=istraining, resize_side_min=_RESIZE_SIDE_MIN), input_data_tensor)

    return input_data_tensor, alpha_tensor
