import tensorflow as tf
import numpy as np
from utils.preprocessing_utils import *

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512
_mean_image = np.load('models/c3d/crop_mean.npy')


def resample_input(video, sample_dims, frame_count, alpha):
    """Return video sampled at uniform rate
    Args:
        :video:       Raw input data
        :frame_count: Total number of frames
        :sample_dims: Number of frames to be provided as input to model
        :alpha        relative sampling rate
    Return:
        Sampled video
    """

    sample_dims = tf.cast(sample_dims, tf.float32)
    indices = tf.range(start=0., limit=sample_dims, delta=1., dtype=tf.float32)
    r_alpha = alpha * tf.cast(frame_count, tf.float32) / sample_dims
    indices = tf.multiply(tf.tile([r_alpha], [tf.cast(sample_dims, tf.int32)]), indices)
    indices = tf.clip_by_value(indices, 0., tf.cast(frame_count-1, tf.float32))
    indices = tf.cast(indices, tf.int32)
    output  = tf.gather(video, tf.convert_to_tensor(indices))
    return output


def preprocess_for_train(image, output_height, output_width, resize_side):
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
  return image


def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
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
                             resize_side_min)
  else:
      return preprocess_for_eval(image, output_height, output_width,
                             resize_side_min)




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

    input_data_tensor = input_data_tensor[...,::-1]

    num_frames_per_clip = input_dims

    # Selecting a random, seeded temporal offset
#    temporal_offset = tf.random_uniform(dtype=tf.int32, minval=0, maxval=frames-num_frames_per_clip, shape=np.asarray([1]))[0]
#    input_data_tensor = input_data_tensor[temporal_offset:temporal_offset+num_frames_per_clip,:,:,:]

    input_data_tensor = tf.cast(input_data_tensor, tf.float32)

    input_data_tensor = resample_input(input_data_tensor, frames, frames, input_alpha)

    input_data_tensor = tf.map_fn(lambda img: preprocess_image(img, size[0], size[1], is_training=istraining, resize_side_min=size[0]), input_data_tensor)

    input_data_tensor = input_data_tensor - _mean_image[...,::-1].tolist()

    return input_data_tensor
