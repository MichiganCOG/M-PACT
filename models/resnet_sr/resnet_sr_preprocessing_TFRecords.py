from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy      as np
from utils.preprocessing_utils import *

#slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
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
  resize_side = tf.random_uniform(
      [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

  image = aspect_preserving_resize(image, resize_side_min)
  image = central_crop([image], output_height, output_width)[0]
  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  #image = tf.image.random_flip_left_right(image)
  return mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


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
  return mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])



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
                                resize_side_min, resize_side_max)
  else:
    return preprocess_for_eval(image, output_height, output_width,
                               resize_side_min)






def model_resample_sinusoidal(video, frame_count, sample_dims, tracker):
	alpha = 1.6
	upper_limit = 3.0
	lower_limit = 0.2
	#indices = tf.range(start=1., limit=float(sample_dims)+1., delta=1., dtype=tf.float32)
	# to ensure one at the last video
	#reduction_factor = tf.asin(tf.cast((1.0 - alpha)*2/float(upper_limit - lower_limit),tf.float32))*4/tf.cast(tf.multiply(number_of_videos, number_of_epochs),tf.float32)
	# Sinusoidal variation with alpha being the DC offset
	#reduction_factor = tf.asin(tf.cast((1.0 - alpha)*2.0/float(upper_limit - lower_limit),tf.float32))*4.0/tf.cast(tf.multiply(number_of_videos, number_of_epochs),tf.float32)
	indices = tf.range(start=0., limit=float(sample_dims), delta=1., dtype=tf.float32)

	# Sinusoidal variation with alpha being the DC offset
	r_alpha = (alpha + (upper_limit - lower_limit) / 2.0 * tf.sin(tf.cast(tracker,tf.float32))) * tf.cast(frame_count, tf.float32) / float(sample_dims)
	#r_alpha = (alpha + (upper_limit - lower_limit) / 2.0 * tf.sin(tf.multiply(tf.cast(tracker,tf.float32),reduction_factor))) * tf.cast(frame_count, tf.float32) / float(sample_dims)
	indices = tf.multiply(tf.tile([r_alpha], [int(sample_dims)]), indices)
	indices = tf.clip_by_value(indices, 0., tf.cast(frame_count-1, tf.float32))
	indices = tf.cast(indices, tf.int32)
	output = tf.gather(video, tf.convert_to_tensor(indices))

	return output, (alpha + (upper_limit - lower_limit) / 2.0 * tf.sin(tf.cast(tracker,tf.float32)))
def resample_input(video, frame_count, sample_dims, alpha):
    """Return video sampled at uniform rate
    Args:
        :video:       Raw input data
        :frame_count: Total number of frames
        :sample_dims: Number of frames to be provided as input to model
        :alpha        relative sampling rate
    Return:
        Sampled video
    """

    indices = tf.range(start=0., limit=float(sample_dims), delta=1., dtype=tf.float32)
    r_alpha = alpha * tf.cast(frame_count, tf.float32) / float(sample_dims)
    indices = tf.multiply(tf.tile([r_alpha], [int(sample_dims)]), indices)
    indices = tf.clip_by_value(indices, 0., tf.cast(frame_count-1, tf.float32))
    indices = tf.cast(indices, tf.int32)
    output  = tf.gather(video, tf.convert_to_tensor(indices))

    return output



def preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label,cvr, input_alpha, istraining, tracker):
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
	footprint = 250
	sample_dims = input_dims

	# END IF

	# Selecting a random, seeded temporal offset
	temporal_offset = tf.random_uniform(dtype=tf.int32, minval=0, maxval=frames-1, shape=np.asarray([1]))[0]

	# get number of videos loaded into the network
	#tracker = tf.get_default_graph().get_tensor_by_name("global_step:0")
	#number_of_videos = [v for v in tf.global_variables() if v.name == 'my_scope/number_of_videos:0'][0]
	#number_of_epochs =  [v for v in tf.global_variables() if v.name == 'my_scope/number_of_epochs:0'][0]
	# Loop video video if it is shorter than footprint
	input_data_tensor =  loop_video_with_offset(input_data_tensor[temporal_offset:,:,:,:], input_data_tensor, frames-temporal_offset, frames, height, width, channel, footprint)
	input_data_tensor = tf.slice(input_data_tensor, [0,0,0,0], tf.stack([footprint, height, width, channel]))
	input_data_tensor = tf.reshape(input_data_tensor, tf.stack([footprint, height, width, channel]))

	if istraining:
		# Reduce footprint to sample_dims in size by sampling rate as sinusoidal function
		input_data_tensor, alpha_tensor = model_resample_sinusoidal(input_data_tensor, footprint, sample_dims, tracker)
	else:
		# Reduce footprint to sample_dims in size by uniform sampling rate
		input_data_tensor = resample_input(input_data_tensor, footprint, sample_dims, input_alpha) #_sample_video(input_data_tensor, footprint, int(footprint/sample_dims),alpha)
		alpha_tensor = 1.0
	input_data_tensor = tf.cast(input_data_tensor, tf.float32)
	input_data_tensor = tf.map_fn(lambda img: preprocess_image(img, size[0], size[1], is_training=istraining,resize_side_min=_RESIZE_SIDE_MIN), input_data_tensor)


	return input_data_tensor, alpha_tensor
