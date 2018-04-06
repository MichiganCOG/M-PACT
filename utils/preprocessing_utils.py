from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


'''

Common preprocessing utilites for input videos. Includes central cropping, random cropping, aspect preserving resizes etc.

'''

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.
The preprocessing steps for VGG were introduced in the following technical
report:
  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0
More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

import tensorflow as tf
import numpy      as np

#slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

RESIZE_SIDE_MIN = 256
RESIZE_SIDE_MAX = 512


def crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.
  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.
  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.
  Returns:
    the cropped (and resized) image.
  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.
  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:
    image, depths, normals = random_crop([image, depths, normals], 120, 150)
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.
  Returns:
    the image_list with cropped images.
  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.
  For example:
    means = [123.68, 116.779, 103.939]
    image = mean_image_subtraction(image, means)
  Note that the rank of `image` must be known.
  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
  Returns:
    the centered image.
  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.
  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width

def largest_size_at_least(height, width, largest_side):
  """Computes new shape with the largest side equal to `largest_side`.
  Computes new shape with the smallest side equal to `largest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    largest_side: A python integer or scalar `Tensor` indicating the size of
      the largest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  largest_side = tf.convert_to_tensor(largest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width  = tf.to_float(width)
  largest_side = tf.to_float(largest_side)

  scale = tf.cond(tf.less(height, width),
                  lambda: largest_side / width,
                  lambda: largest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)

  return new_height, new_width

def aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=True)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image

def aspect_preserving_resize_largest(image, largest_side):
  """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image `Tensor`.
    largest_side: A python integer or scalar `Tensor` indicating the size of
      the largest side after resize.
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  largest_side = tf.convert_to_tensor(largest_side, dtype=tf.int32)

  shape  = tf.shape(image)
  height = shape[0]
  width  = shape[1]

  new_height, new_width = largest_size_at_least(height, width, largest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=True)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])

  return resized_image

def loop_video_with_offset(offset_tensor, input_data_tensor, offset_frames, frames, height, width, channel, footprint):
    """
    Loop the video the number of times necessary for the number of frames to be > footprint
    Args:
        :offset_tensor:     Raw input data from offset frame number
        :input_data_tensor: Raw input data
        :frames:            Total number of frames
        :height:            Height of frame
        :width:             Width of frame
        :channel:           Total number of color channels
        :footprint:         Total length of video to be extracted before sampling down

    Return:
        Looped video
    """

    loop_factor       = tf.cast(tf.add(tf.divide(tf.subtract(footprint, offset_frames), frames), 1), tf.int32)
    loop_stack        = tf.stack([loop_factor,1,1,1])
    input_data_tensor = tf.tile(input_data_tensor, loop_stack)
    reshape_stack     = tf.stack([tf.multiply(frames, loop_factor),height,width,channel])
    input_data_looped = tf.reshape(input_data_tensor, reshape_stack)

    output_data       = tf.concat([offset_tensor, input_data_looped], axis = 0)

    return output_data



def resample_input(video, sample_dims, frame_count, alpha):
    """Return video sampled at uniform rate
    Args:
        :video:       Raw input data
        :sample_dims: Number of frames to be provided as input to model
        :frame_count: Total number of frames
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

def resample_model(video, sample_dims, frame_count, alpha):
    """Return video sampled at uniform rate
    Args:
        :video:       Raw input data
        :sample_dims: Number of frames to be provided as input to model
        :frame_count: Total number of frames
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


def resample_model_sinusoidal(video, sample_dims, frame_count, tracker):
    """Return video sampled to a rate chosen sinuloidally based on the current accumulated number of videos.
       Resampling factors are chosen from a range of 0.2 to 3.0.
    Args:
        :video:       Raw input data
        :sample_dims: Number of frames to be provided as input to model
        :frame_count: Total number of frames
        :tracker:     Number of videos that have been loaded in total during training
    Return:
        Sampled video
    """
    alpha       = 1.6
    upper_limit = 3.0
    lower_limit = 0.2

    indices = tf.range(start=0., limit=float(sample_dims), delta=1., dtype=tf.float32)

    # Sinusoidal variation with alpha being the DC offset
    r_alpha = (alpha + (upper_limit - lower_limit) / 2.0 * tf.sin(tf.cast(tracker,tf.float32))) * tf.cast(frame_count, tf.float32) / float(sample_dims)

    indices = tf.multiply(tf.tile([r_alpha], [int(sample_dims)]), indices)
    indices = tf.clip_by_value(indices, 0., tf.cast(frame_count-1, tf.float32))

    indices = tf.cast(indices, tf.int32)
    output  = tf.gather(video, tf.convert_to_tensor(indices))
    return output, (alpha + (upper_limit - lower_limit) / 2.0 * tf.sin(tf.cast(tracker,tf.float32)))
