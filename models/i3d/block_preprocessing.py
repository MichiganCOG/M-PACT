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

  image = tf.to_float(image)
  image = (image/255.) * 2. - 1.

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

  image = tf.to_float(image)
  image = (image/255.) * 2. - 1.

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

def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, input_alpha):
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
        :input_alpha:       Alpha value to resample input_data_tensor (independent of model)

    Return:
        Preprocessing input data and labels tensor
    """

    """
    Original preprocessing algorithm
        1. Temporal offset, loop if necessary
        2. Slice the video into required size
        3. Resize input frames to desired size
        4. Randomly crop frames or centrally crop frames
        5. Randomly flip the clip
    """

    """
    Block preprocessing algorithm
        1. Divide video into 3 segments
        2. Loop video if number of frames is lower
        3. Sample uniformly from each segment to make up the total number of frames
        4. Resize input frames to desired size
        5. Randomly crop frames or centrally crop frames
        6. Randomly flip the clip
    """


    # Setup different temporal footprints for training and testing phase
    if istraining:
        footprint   = input_dims
        sample_dims = input_dims

    else:
        footprint   = 250
        sample_dims = input_dims

    #Training: input_dims == 64
    #Testing:  input_dims == 79

    # END IF

    ####################################### MODIFIED FROM DEFAULT ###################################################################3
    input_data_tensor = tf.cond(tf.less(frames, footprint),
                                lambda: loop_video_with_offset(input_data_tensor, input_data_tensor, frames, frames, height, width, channel, footprint),
                                lambda: input_data_tensor)

    # If training phase, divide video into 3 segments
    if istraining:
        segment       = 4 
        current_frames = tf.shape(input_data_tensor)[0]

        segment_1_idx = tf.linspace(start = 0., stop = tf.cast(current_frames / segment , tf.float32)- tf.constant(1.), num = tf.cast(footprint / segment, tf.int32))

        segment_2_idx = tf.linspace(start = tf.cast(current_frames/segment, tf.float32), stop = tf.cast((tf.constant(2)* current_frames / segment), tf.float32) - tf.constant(1.), num = tf.cast(footprint / segment, tf.int32))
        segment_3_idx = tf.linspace(start = tf.constant(2.)*tf.cast((current_frames/segment),tf.float32), stop = tf.cast((tf.constant(3)*current_frames / segment), tf.float32) - tf.constant(1.), num = tf.cast(footprint / segment, tf.int32))
        segment_4_idx = tf.linspace(start = tf.constant(3.)*tf.cast((current_frames/segment),tf.float32), stop = tf.cast((tf.constant(4)*current_frames / segment),tf.float32) - tf.constant(1.), num = tf.cast(footprint / segment, tf.int32))

        input_data_tensor = tf.concat([tf.gather(input_data_tensor, tf.cast(segment_1_idx, tf.int32)), tf.gather(input_data_tensor, tf.cast(segment_2_idx, tf.int32)), tf.gather(input_data_tensor, tf.cast(segment_3_idx, tf.int32)), tf.gather(input_data_tensor, tf.cast(segment_4_idx, tf.int32))], 0)


    else:
        input_data_tensor = tf.slice(input_data_tensor, [0,0,0,0], tf.stack([footprint, height, width, channel]))
        input_data_tensor = resample_input(input_data_tensor, sample_dims, footprint, 1.0)

    ####################################### MODIFIED FROM DEFAULT ###################################################################3

    input_data_tensor = tf.reshape(input_data_tensor, tf.stack([footprint, height, width, channel]))
    input_data_tensor = tf.cast(input_data_tensor, tf.float32)

    # Randomly flip entire video or not
    crop_type = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=np.asarray([1]))[0]

    # Preprocess data
    input_data_tensor = tf.map_fn(lambda img: preprocess_image(img, size[0], size[1], is_training=istraining, resize_side_min=_RESIZE_SIDE_MIN), input_data_tensor)

    if istraining:
        input_data_tensor = tf.cond(tf.greater_equal(crop_type, 0.5), lambda: random_crop_clip(input_data_tensor, size[0], size[1]), lambda: central_crop_clip(input_data_tensor, size[0], size[1]))
        input_data_tensor = random_flip_left_right_clip(input_data_tensor)

    else:
        input_data_tensor = central_crop_clip(input_data_tensor, size[0], size[1])

    # END IF

    input_data_tensor = tf.reshape(input_data_tensor, tf.stack([footprint, size[0], size[1], channel]))

    return input_data_tensor
